"""
Custom solver for image generation tasks.

This module provides a solver that queries image generation APIs (OpenAI GPT-Image/DALL-E
or Google Nano Banana) to generate images based on text prompts. Since Inspect AI doesn't
natively support image generation models, this custom solver handles the API calls
and image storage.
"""

import asyncio
import base64
import os
import uuid
from typing import Optional

from inspect_ai.model import (
    ChatMessageAssistant,
    ContentImage,
    ContentText,
    ModelOutput,
)
from inspect_ai.solver import Solver, TaskState, solver

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


def save_base64_image(
    base64_data: str,
    output_dir: str,
    sample_id: str,
    format: str = "png",
) -> str:
    """
    Save a base64-encoded image to a file.

    Args:
        base64_data: The base64-encoded image data.
        output_dir: Directory to save the image.
        sample_id: Sample ID for naming the file.
        format: Image format (png, jpeg, webp).

    Returns:
        The absolute path to the saved image.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate unique filename
    filename = f"generated_{sample_id}_{uuid.uuid4().hex[:8]}.{format}"
    filepath = os.path.join(output_dir, filename)

    # Decode and save
    image_bytes = base64.b64decode(base64_data)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return os.path.abspath(filepath)


def pil_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


async def generate_image_openai(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "auto",
    n: int = 1,
    output_format: str = "png",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Generate an image using OpenAI's Image API (DALL-E or GPT-Image).

    Args:
        prompt: Text prompt for image generation.
        model: Model to use.
        size: Image size (1024x1024, 1536x1024, 1024x1536).
        quality: Quality setting (low, medium, high, auto).
        n: Number of images to generate.
        output_format: Output format (png, jpeg, webp).
        api_key: OpenAI API key (uses env var if not provided).
        base_url: Custom base URL for API.

    Returns:
        Tuple of (base64 image data, usage metadata dict).
    """
    if AsyncOpenAI is None:
        raise ImportError("openai package is required for OpenAI image generation")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Check if this is a GPT image model or a DALL-E model
    is_gpt_image_model = model.startswith("gpt-image")
    is_dalle3 = model == "dall-e-3"
    is_dalle2 = model == "dall-e-2"

    if is_gpt_image_model:
        # GPT image models use output_format instead of response_format
        # They always return base64-encoded images
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            output_format=output_format,  # png, jpeg, or webp
        )
    elif is_dalle3:
        # DALL-E 3 supports quality (hd or standard) and response_format
        dalle3_quality = "hd" if quality in ["high", "hd"] else "standard"
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=dalle3_quality,
            n=n,
            response_format="b64_json",
        )
    else:
        # DALL-E 2 doesn't support quality parameter at all
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            response_format="b64_json",
        )

    # Extract image data
    image_data = response.data[0].b64_json
    revised_prompt = getattr(response.data[0], "revised_prompt", None)

    metadata = {
        "model": model,
        "revised_prompt": revised_prompt,
        "size": size,
        "quality": quality if not is_dalle2 else "standard",
        "usage": (
            response.usage.model_dump(mode="json")
            if hasattr(response, "usage") and response.usage
            else None
        ),
    }

    return image_data, metadata


async def generate_image_google(
    prompt: str,
    model: str = "gemini-2.5-flash-image",
    sample_count: int = 1,
    aspect_ratio: str = "1:1",
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    location: str = "us-central1",
    input_images: Optional[list[str]] = None,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> tuple[str, dict]:
    """
    Generate an image using Google's Gemini or Imagen API.
    Supports two types of models:
    - Gemini image models (gemini-2.5-flash-image, gemini-3-pro-image-preview):
    - Imagen models (imagen-4.0-generate-001, etc.):

    Args:
        prompt: Text prompt for image generation.
        model: Model to use.
        sample_count: Number of images to generate.
        aspect_ratio: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, etc.).
        api_key: Google API key (uses env var if not provided).
        project_id: Google Cloud project ID (for Imagen).
        location: Google Cloud location (for Imagen).
        input_images: Optional list of input image paths/URLs for image-to-image tasks.
        max_retries: Maximum number of retry attempts for NO_IMAGE responses (default: 3).
        retry_delay: Delay in seconds between retries (default: 10.0).

    Returns:
        Tuple of (base64 image data, usage metadata dict).
    """
    if genai is None:
        raise ImportError(
            "google-genai package is required for Google image generation"
        )

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Check if this is a Gemini image model or an Imagen model
    is_gemini_image_model = model.startswith("gemini-") and "image" in model.lower()

    if is_gemini_image_model:
        # Build contents list with optional input images
        contents = []

        # Add input images first if provided (for image-to-image tasks)
        if input_images:
            for img in input_images:
                # Handle both file paths and data URLs
                if img.startswith("data:"):
                    # Extract base64 from data URL
                    parts = img.split(",", 1)
                    if len(parts) == 2:
                        mime_match = (
                            parts[0].split(";")[0].split(":")[1]
                            if ":" in parts[0]
                            else "image/png"
                        )
                        contents.append(
                            genai_types.Part.from_bytes(
                                data=base64.b64decode(parts[1]),
                                mime_type=mime_match,
                            )
                        )
                elif os.path.exists(img):
                    # Local file path
                    with open(img, "rb") as f:
                        img_bytes = f.read()
                    mime_type = "image/png" if img.endswith(".png") else "image/jpeg"
                    contents.append(
                        genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                    )
                else:
                    # Assume it's a URL - let the API handle it
                    contents.append(
                        genai_types.Part.from_uri(file_uri=img, mime_type="image/jpeg")
                    )

        contents.append(prompt)

        # Retry loop
        last_error = None
        for attempt in range(max_retries):
            # Use generate_content API for Gemini image models
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=genai_types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

            # Validate response
            if response is None:
                raise ValueError("Gemini API returned None response")
            if not response.candidates or len(response.candidates) == 0:
                raise ValueError("Gemini API returned no candidates")

            finish_reason = response.candidates[0].finish_reason

            # Check for NO_IMAGE finish reason - this is retriable
            if response.candidates[0].content is None:
                if "NO_IMAGE" in str(finish_reason):
                    last_error = ValueError(
                        f"Gemini declined to generate an image (finish_reason: {finish_reason}). "
                        f"This may happen when the prompt is interpreted as a question rather than "
                        f"an image generation request. Original prompt: {prompt[:200]}..."
                    )
                    if attempt < max_retries - 1:
                        # Wait before retrying
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise last_error
                raise ValueError(
                    f"Gemini API returned no content. Finish reason: {finish_reason}"
                )
            if not response.candidates[0].content.parts:
                if "NO_IMAGE" in str(finish_reason):
                    last_error = ValueError(
                        f"Gemini returned no image parts (finish_reason: {finish_reason}). "
                        f"Attempt {attempt + 1}/{max_retries}."
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise last_error
                raise ValueError(
                    f"Gemini API returned no parts in content. Finish reason: {finish_reason}"
                )

            # Successful response with image
            break

        # Extract image from response parts
        image_base64 = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                # Image data is in inline_data.data (bytes)
                image_bytes = part.inline_data.data
                image_base64 = base64.b64encode(image_bytes).decode()
                break

        if image_base64 is None:
            raise ValueError("No image generated by Gemini model")

        metadata = {
            "model": model,
            "aspect_ratio": aspect_ratio,
            "model_type": "gemini_image",
            "usage": (
                response.usage_metadata.model_dump(mode="json")
                if response.usage_metadata
                else None
            ),
        }
    else:
        # Use generate_images API for Imagen models
        response = await client.aio.models.generate_images(
            model=model,
            prompt=prompt,
            config=genai_types.GenerateImagesConfig(
                number_of_images=sample_count,
                aspect_ratio=aspect_ratio,
            ),
        )

        # Extract image data
        if response.generated_images and len(response.generated_images) > 0:
            generated_image = response.generated_images[0]
            # The image is returned as PIL Image, convert to base64
            image_base64 = pil_image_to_base64(generated_image.image)
        else:
            raise ValueError("No image generated by Google Imagen API")

        metadata = {
            "model": model,
            "aspect_ratio": aspect_ratio,
            "model_type": "imagen",
            # usage not available for imagen
        }

    return image_base64, metadata


@solver
def image_generation_solver(
    backend: str = "openai",
    model_name: str = "gpt-image-1",
    output_dir: str = "./outputs/generated_images",
    size: str = "1024x1024",
    quality: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> Solver:
    """
    Custom solver for image generation tasks.

    This solver takes the input prompt and generates an image using the specified
    backend (OpenAI or Google). The generated image is saved to disk and its path
    is stored in the task state.

    Args:
        backend: Image generation backend ("openai" or "google").
        model_name: Model to use for image generation.
        output_dir: Directory to save generated images.
        size: Image size (for OpenAI: 1024x1024, 1536x1024, 1024x1536).
        quality: Quality setting (for OpenAI: low, medium, high, auto).
        api_key: API key (uses environment variable if not provided).
        base_url: Custom base URL for API (OpenAI only).
        **kwargs: Additional arguments passed to the generation function.

    Returns:
        Solver function that generates images.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        """Generate an image based on the input prompt."""

        # Extract the text prompt from the input
        raw_prompt = state.input_text

        # Extract any input images from the original question
        # These are needed for image-to-image generation tasks
        input_images = []
        for msg in state.messages:
            if hasattr(msg, "content") and isinstance(msg.content, list):
                for content in msg.content:
                    if isinstance(content, ContentImage):
                        input_images.append(content.image)

        # Wrap the prompt in an explicit image generation instruction
        # This helps Gemini understand that we want an image, not a text answer
        # The model may refuse to generate images if the prompt looks like a question
        if input_images:
            # For image-to-image tasks, be more specific
            image_generation_prompt = (
                f"Based on the provided input image(s), generate a new image that "
                f"addresses the following request:\n\n"
                f"{raw_prompt}\n\n"
                f"Create a clear, detailed image as your response."
            )
        else:
            image_generation_prompt = (
                f"Generate an image that answers or addresses the following:\n\n"
                f"{raw_prompt}\n\n"
                f"Create a clear, detailed image as your response."
            )

        # Get sample ID for filename
        sample_id = str(state.sample_id) if state.sample_id else "unknown"

        try:
            # Generate image based on backend
            if backend == "openai":
                image_base64, metadata = await generate_image_openai(
                    prompt=image_generation_prompt,
                    model=model_name,
                    size=size,
                    quality=quality,
                    api_key=api_key,
                    base_url=base_url,
                )
                output_format = "png"
            elif backend == "google":
                # Map size to aspect ratio for Google
                aspect_ratio_map = {
                    "1024x1024": "1:1",
                    "1536x1024": "3:2",
                    "1024x1536": "2:3",
                }
                aspect_ratio = aspect_ratio_map.get(size, "1:1")

                image_base64, metadata = await generate_image_google(
                    prompt=image_generation_prompt,
                    model=model_name,
                    aspect_ratio=aspect_ratio,
                    api_key=api_key,
                    input_images=input_images if input_images else None,
                    **kwargs,
                )
                output_format = "png"
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            # Save image to disk
            image_path = save_base64_image(
                base64_data=image_base64,
                output_dir=output_dir,
                sample_id=sample_id,
                format=output_format,
            )

            # Store the generated image path and base64 data in state
            # The grader will need access to this
            state.store.set("generated_image_path", image_path)
            state.store.set("generated_image_base64", image_base64)
            state.store.set("image_generation_metadata", metadata)

            # Create a response message indicating image was generated
            # Include the image path in the completion for the grader
            completion_text = f"[Generated Image: {image_path}]"

            # Create data URL for proper visualization in Inspect UI
            # Determine mime type based on format
            mime_type = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }.get(output_format, "image/png")
            image_data_url = f"data:{mime_type};base64,{image_base64}"

            # Store the data URL for grader and UI display
            state.store.set("generated_image_data_url", image_data_url)

            # Create output with the image information
            state.output = ModelOutput.from_content(
                model=f"{backend}/{model_name}",
                content=[
                    ContentText(text=completion_text),
                    ContentImage(image=image_data_url),
                ],
            )

            # Also add to messages for context
            state.messages.append(
                ChatMessageAssistant(
                    content=[
                        ContentText(text=completion_text),
                        ContentImage(image=image_data_url),
                    ]
                )
            )

        except Exception as e:
            # Handle generation errors gracefully
            error_message = f"Image generation failed: {str(e)}"
            state.output = ModelOutput.from_content(
                model=f"{backend}/{model_name}",
                content=error_message,
            )
            state.messages.append(ChatMessageAssistant(content=error_message))
            state.store.set("generation_error", str(e))

        return state

    return solve


def get_image_generation_solver(solver_config: dict) -> Solver:
    """
    Factory function to create an image generation solver from configuration.

    Args:
        solver_config: Configuration dictionary containing:
            - backend: "openai" or "google"
            - model_name: Model name for image generation
            - output_dir: Directory to save generated images
            - size: Image size
            - quality: Quality setting
            - api_key: Optional API key
            - base_url: Optional base URL

    Returns:
        Configured image generation solver.
    """
    if "backend" not in solver_config or "model_name" not in solver_config:
        raise ValueError("solver_config must contain 'backend' and 'model_name' keys.")

    return image_generation_solver(
        backend=solver_config["backend"],
        model_name=solver_config["model_name"],
        output_dir=solver_config.get("output_dir", "./outputs/generated_images"),
        size=solver_config.get("size", "1024x1024"),
        quality=solver_config.get("quality", "auto"),
        api_key=solver_config.get("api_key"),
        base_url=solver_config.get("base_url"),
    )
