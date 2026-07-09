# Reasoning Blind Spots Benchmark

> **`blind-spots-bench`** is a benchmark designed to stress test the reasoning capabilities of frontier AI models on tasks that are straightforward for humans but difficult for AI.
The questions are crafted to highlight the limitations of current AI systems and understand where they struggle in reasoning tasks. Download the dataset from [HuggingFace](https://huggingface.co/datasets/matsant01/blind-spots-bench) 🤗.

Our codebase relies on [Inspect AI](https://inspect.aisi.org.uk) as the evaluation framework.

---

### Results


<div align="left">
  <img src="notebooks/assets/front_page_new.png" width="800">

</div>


<details>
<summary> 🏆 Complete Scores </summary>
<div>

<table aria-label="LLM and VLM model results by evaluation subset">
    <thead>
      <tr>
        <th></th>
        <th align="center" colspan="4">Text-only</th>
        <th align="center" colspan="4">Multi-to-text</th>
      </tr>
      <tr>
        <th align="left">Model</th>
        <th>mean@4 (%)</th>
        <th>pass@4 (%)</th>
        <th>out-tks</th>
        <th>cost (&#36;/100)</th>
        <th>mean@4 (%)</th>
        <th>pass@4 (%)</th>
        <th>out-tks</th>
        <th>cost (&#36;/100)</th>
      </tr>
    </thead>
    <tbody>
      <tr><th align="left" colspan="9">Text-only models</th></tr>
      <tr>
<th align="left">GLM-4.7</th>
<td align="right" bgcolor="#dde2bf" style="background-color:#dde2bf;">61.3<sub>± 4.1</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">75.9<sub>± 4.1</sub>%</td>
<td align="right" bgcolor="#e4dabf" style="background-color:#e4dabf;">5346<sub>± 419</sub></td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">0.91</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">GLM-5</th>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">64.4<sub>± 4</sub>%</td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">77.8<sub>± 4</sub>%</td>
<td align="right" bgcolor="#d8e6bf" style="background-color:#d8e6bf;">3404<sub>± 216</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">1.11</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">GLM-5.1</th>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">67.4<sub>± 4.1</sub></td>
<td align="right" bgcolor="#d6e9bf" style="background-color:#d6e9bf;">76.9<sub>± 4.1</sub></td>
<td align="right" bgcolor="#d6e9bf" style="background-color:#d6e9bf;">2948<sub>± 250</sub></td>
<td align="right" bgcolor="#d2edbf" style="background-color:#d2edbf;">0.69</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">GLM-5.2</th>
<td align="right" bgcolor="#d2edbf" style="background-color:#d2edbf;">73.8<sub>± 3.6</sub></td>
<td align="right" bgcolor="#cff0bf" style="background-color:#cff0bf;">84.3<sub>± 3.5</sub></td>
<td align="right" bgcolor="#e4dbbf" style="background-color:#e4dbbf;">5229<sub>± 486</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">1.09</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">DeepSeek-V4-Flash</th>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">67.6<sub>± 3.9</sub></td>
<td align="right" bgcolor="#d3ebbf" style="background-color:#d3ebbf;">79.6<sub>± 3.9</sub></td>
<td align="right" bgcolor="#d9e5bf" style="background-color:#d9e5bf;">3648<sub>± 391</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">0.29</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">DeepSeek-V4-Pro</th>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">70.4<sub>± 3.8</sub></td>
<td align="right" bgcolor="#d2ecbf" style="background-color:#d2ecbf;">80.6<sub>± 3.8</sub></td>
<td align="right" bgcolor="#d1edbf" style="background-color:#d1edbf;">2232<sub>± 201</sub></td>
<td align="right" bgcolor="#d1edbf" style="background-color:#d1edbf;">0.63</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">GPT-oss-20b</th>
<td align="right" bgcolor="#e4dabf" style="background-color:#e4dabf;">52.3<sub>± 4.1</sub></td>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">67.6<sub>± 4.5</sub></td>
<td align="right" bgcolor="#d3ebbf" style="background-color:#d3ebbf;">2611<sub>± 317</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.02</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">GPT-oss-120b</th>
<td align="right" bgcolor="#dde1bf" style="background-color:#dde1bf;">60.2<sub>± 4.1</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">72.2<sub>± 4.3</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">1555<sub>± 204</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.02</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr>
<th align="left">Qwen3-Next-80B-A3B</th>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">59.5<sub>± 4.2</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">72.2<sub>± 4.3</sub></td>
<td align="right" bgcolor="#ebd3bf" style="background-color:#ebd3bf;">6460<sub>± 444</sub></td>
<td align="right" bgcolor="#ccf2bf" style="background-color:#ccf2bf;">0.27</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
      <tr><th align="left" colspan="9">VLMs</th></tr>
      <tr>
<th align="left">Kimi-K2.5</th>
<td align="right" bgcolor="#d6e8bf" style="background-color:#d6e8bf;">68.3<sub>± 3.8</sub></td>
<td align="right" bgcolor="#d2ecbf" style="background-color:#d2ecbf;">80.6<sub>± 3.8</sub></td>
<td align="right" bgcolor="#dde2bf" style="background-color:#dde2bf;">4163<sub>± 336</sub></td>
<td align="right" bgcolor="#d4eabf" style="background-color:#d4eabf;">0.86</td>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">42.4<sub>± 6.8</sub></td>
<td align="right" bgcolor="#e2dcbf" style="background-color:#e2dcbf;">53.5<sub>± 7.7</sub></td>
<td align="right" bgcolor="#e9d5bf" style="background-color:#e9d5bf;">4994<sub>± 690</sub></td>
<td align="right" bgcolor="#d3ebbf" style="background-color:#d3ebbf;">1.07</td>
</tr>
      <tr>
<th align="left">Kimi-K2.6</th>
<td align="right" bgcolor="#d9e5bf" style="background-color:#d9e5bf;">65<sub>± 3.9</sub></td>
<td align="right" bgcolor="#d4ebbf" style="background-color:#d4ebbf;">78.7<sub>± 4</sub></td>
<td align="right" bgcolor="#dbe4bf" style="background-color:#dbe4bf;">3797<sub>± 266</sub></td>
<td align="right" bgcolor="#d2ecbf" style="background-color:#d2ecbf;">0.7</td>
<td align="right" bgcolor="#e0debf" style="background-color:#e0debf;">41.9<sub>± 6.8</sub></td>
<td align="right" bgcolor="#e2dcbf" style="background-color:#e2dcbf;">53.5<sub>± 7.7</sub></td>
<td align="right" bgcolor="#e6d9bf" style="background-color:#e6d9bf;">4501<sub>± 527</sub></td>
<td align="right" bgcolor="#d1edbf" style="background-color:#d1edbf;">0.87</td>
</tr>
      <tr>
<th align="left">Gemma-4-E2B-it</th>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">32.4<sub>± 4</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">42.6<sub>± 4.8</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">1533<sub>± 127</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.01</td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">18<sub>± 4.7</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">30.2<sub>± 7.1</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">1049<sub>± 139</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.01</td>
</tr>
      <tr>
<th align="left">Gemma-4-E4B-it</th>
<td align="right" bgcolor="#eed0bf" style="background-color:#eed0bf;">41.4<sub>± 4</sub></td>
<td align="right" bgcolor="#e7d7bf" style="background-color:#e7d7bf;">57.4<sub>± 4.8</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">1601<sub>± 147</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.01</td>
<td align="right" bgcolor="#f1cdbf" style="background-color:#f1cdbf;">22.7<sub>± 5.8</sub></td>
<td align="right" bgcolor="#f4cbbf" style="background-color:#f4cbbf;">32.6<sub>± 7.2</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">1335<sub>± 287</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.01</td>
</tr>
      <tr>
<th align="left">Gemma-4-26B-A4B-it</th>
<td align="right" bgcolor="#dbe3bf" style="background-color:#dbe3bf;">62.5<sub>± 4.2</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">72.2<sub>± 4.3</sub></td>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">4541<sub>± 341</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.06</td>
<td align="right" bgcolor="#e0debf" style="background-color:#e0debf;">40.7<sub>± 6.9</sub></td>
<td align="right" bgcolor="#e2dcbf" style="background-color:#e2dcbf;">53.5<sub>± 7.7</sub></td>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">3587<sub>± 578</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.05</td>
</tr>
      <tr>
<th align="left">Qwen3-VL-30B-A3B</th>
<td align="right" bgcolor="#e7d7bf" style="background-color:#e7d7bf;">49.5<sub>± 4.1</sub></td>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">66.7<sub>± 4.6</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">3673<sub>± 256</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.07</td>
<td align="right" bgcolor="#efcfbf" style="background-color:#efcfbf;">25<sub>± 6</sub></td>
<td align="right" bgcolor="#f0cfbf" style="background-color:#f0cfbf;">37.2<sub>± 7.5</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">2796<sub>± 538</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.06</td>
</tr>
      <tr>
<th align="left">Qwen3-VL-235B-A22B</th>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">58.3<sub>± 4.2</sub></td>
<td align="right" bgcolor="#dbe4bf" style="background-color:#dbe4bf;">71.3<sub>± 4.4</sub></td>
<td align="right" bgcolor="#d8e6bf" style="background-color:#d8e6bf;">3435<sub>± 222</sub></td>
<td align="right" bgcolor="#cdf1bf" style="background-color:#cdf1bf;">0.32</td>
<td align="right" bgcolor="#e9d6bf" style="background-color:#e9d6bf;">32<sub>± 6.8</sub></td>
<td align="right" bgcolor="#eed0bf" style="background-color:#eed0bf;">39.5<sub>± 7.5</sub></td>
<td align="right" bgcolor="#d6e9bf" style="background-color:#d6e9bf;">2604<sub>± 351</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">0.26</td>
</tr>
      <tr>
<th align="left">Qwen3.5-35B-A3B</th>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">63.9<sub>± 4.2</sub></td>
<td align="right" bgcolor="#d8e6bf" style="background-color:#d8e6bf;">74.1<sub>± 4.2</sub></td>
<td align="right" bgcolor="#f0cebf" style="background-color:#f0cebf;">7278<sub>± 374</sub></td>
<td align="right" bgcolor="#caf4bf" style="background-color:#caf4bf;">0.11</td>
<td align="right" bgcolor="#dbe4bf" style="background-color:#dbe4bf;">47.1<sub>± 6.5</sub></td>
<td align="right" bgcolor="#d4eabf" style="background-color:#d4eabf;">69.8<sub>± 7.1</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">6492<sub>± 643</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">0.1</td>
</tr>
      <tr>
<th align="left">Qwen3.5-122B-A10B</th>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">63.9<sub>± 4.2</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">75<sub>± 4.2</sub></td>
<td align="right" bgcolor="#e6d8bf" style="background-color:#e6d8bf;">5637<sub>± 284</sub></td>
<td align="right" bgcolor="#ccf2bf" style="background-color:#ccf2bf;">0.25</td>
<td align="right" bgcolor="#d9e5bf" style="background-color:#d9e5bf;">48.8<sub>± 6.9</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">62.8<sub>± 7.5</sub></td>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">3350<sub>± 353</sub></td>
<td align="right" bgcolor="#caf4bf" style="background-color:#caf4bf;">0.16</td>
</tr>
      <tr>
<th align="left">Qwen3.5-397B-A17B</th>
<td align="right" bgcolor="#d4eabf" style="background-color:#d4eabf;">71.1<sub>± 3.9</sub></td>
<td align="right" bgcolor="#d1edbf" style="background-color:#d1edbf;">81.5<sub>± 3.8</sub></td>
<td align="right" bgcolor="#e7d7bf" style="background-color:#e7d7bf;">5849<sub>± 288</sub></td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">0.89</td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">48.3<sub>± 7.2</sub></td>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">58.1<sub>± 7.6</sub></td>
<td align="right" bgcolor="#edd1bf" style="background-color:#edd1bf;">5479<sub>± 502</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">0.85</td>
</tr>
      <tr>
<th align="left">Gemini-2.5-flash</th>
<td align="right" bgcolor="#e2ddbf" style="background-color:#e2ddbf;">55.3<sub>± 4.1</sub></td>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">70.4<sub>± 4.4</sub></td>
<td align="right" bgcolor="#dbe3bf" style="background-color:#dbe3bf;">3884<sub>± 513</sub></td>
<td align="right" bgcolor="#d6e9bf" style="background-color:#d6e9bf;">0.97</td>
<td align="right" bgcolor="#e9d6bf" style="background-color:#e9d6bf;">32<sub>± 6.5</sub></td>
<td align="right" bgcolor="#ecd2bf" style="background-color:#ecd2bf;">41.9<sub>± 7.6</sub></td>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">3383<sub>± 997</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">0.85</td>
</tr>
      <tr>
<th align="left">Gemini-2.5-pro</th>
<td align="right" bgcolor="#dde1bf" style="background-color:#dde1bf;">60.4<sub>± 4</sub></td>
<td align="right" bgcolor="#d6e9bf" style="background-color:#d6e9bf;">76.9<sub>± 4.1</sub></td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">2905<sub>± 311</sub></td>
<td align="right" bgcolor="#f0cfbf" style="background-color:#f0cfbf;">2.92</td>
<td align="right" bgcolor="#e5d9bf" style="background-color:#e5d9bf;">36<sub>± 6.8</sub></td>
<td align="right" bgcolor="#ead4bf" style="background-color:#ead4bf;">44.2<sub>± 7.7</sub></td>
<td align="right" bgcolor="#dbe3bf" style="background-color:#dbe3bf;">3283<sub>± 748</sub></td>
<td align="right" bgcolor="#e7d7bf" style="background-color:#e7d7bf;">3.32</td>
</tr>
      <tr>
<th align="left">Gemini-3-flash</th>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">76.2<sub>± 3.7</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">83.3<sub>± 3.6</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">8130<sub>± 637</sub></td>
<td align="right" bgcolor="#e9d5bf" style="background-color:#e9d5bf;">2.45</td>
<td align="right" bgcolor="#cbf4bf" style="background-color:#cbf4bf;">64.5<sub>± 6.2</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">83.7<sub>± 5.7</sub></td>
<td align="right" bgcolor="#d9e6bf" style="background-color:#d9e6bf;">2942<sub>± 639</sub></td>
<td align="right" bgcolor="#d2edbf" style="background-color:#d2edbf;">0.94</td>
</tr>
      <tr>
<th align="left">Gemini-3.1-flash-lite</th>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">61.8<sub>± 3.9</sub></td>
<td align="right" bgcolor="#d4ebbf" style="background-color:#d4ebbf;">78.7<sub>± 4</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">1537<sub>± 166</sub></td>
<td align="right" bgcolor="#ccf2bf" style="background-color:#ccf2bf;">0.23</td>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">42.4<sub>± 7.1</sub></td>
<td align="right" bgcolor="#e4dabf" style="background-color:#e4dabf;">51.2<sub>± 7.7</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">1335<sub>± 112</sub></td>
<td align="right" bgcolor="#cbf4bf" style="background-color:#cbf4bf;">0.23</td>
</tr>
      <tr>
<th align="left">Gemini-3.1-pro</th>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">83.3<sub>± 3.2</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">90.7<sub>± 2.8</sub></td>
<td align="right" bgcolor="#e6d9bf" style="background-color:#e6d9bf;">5550<sub>± 645</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">3.34</td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">66.9<sub>± 6.4</sub></td>
<td align="right" bgcolor="#cff0bf" style="background-color:#cff0bf;">76.7<sub>± 6.5</sub></td>
<td align="right" bgcolor="#e1ddbf" style="background-color:#e1ddbf;">3959<sub>± 858</sub></td>
<td align="right" bgcolor="#e0debf" style="background-color:#e0debf;">2.49</td>
</tr>
      <tr>
<th align="left">GPT-5-mini</th>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">67.6<sub>± 4</sub></td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">77.8<sub>± 4</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">2032<sub>± 225</sub></td>
<td align="right" bgcolor="#cef0bf" style="background-color:#cef0bf;">0.41</td>
<td align="right" bgcolor="#e7d7bf" style="background-color:#e7d7bf;">33.1<sub>± 6.5</sub></td>
<td align="right" bgcolor="#ead4bf" style="background-color:#ead4bf;">44.2<sub>± 7.7</sub></td>
<td align="right" bgcolor="#ccf2bf" style="background-color:#ccf2bf;">1437<sub>± 221</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">0.3</td>
</tr>
      <tr>
<th align="left">GPT-5</th>
<td align="right" bgcolor="#d4eabf" style="background-color:#d4eabf;">70.6<sub>± 3.8</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">82.4<sub>± 3.7</sub></td>
<td align="right" bgcolor="#d6e8bf" style="background-color:#d6e8bf;">3104<sub>± 329</sub></td>
<td align="right" bgcolor="#f2ccbf" style="background-color:#f2ccbf;">3.12</td>
<td align="right" bgcolor="#e4dbbf" style="background-color:#e4dbbf;">37.8<sub>± 6.7</sub></td>
<td align="right" bgcolor="#e4dabf" style="background-color:#e4dabf;">51.2<sub>± 7.7</sub></td>
<td align="right" bgcolor="#d3ebbf" style="background-color:#d3ebbf;">2279<sub>± 282</sub></td>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">2.35</td>
</tr>
      <tr>
<th align="left">GPT-5.2</th>
<td align="right" bgcolor="#d4eabf" style="background-color:#d4eabf;">70.6<sub>± 3.7</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">83.3<sub>± 3.6</sub></td>
<td align="right" bgcolor="#caf4bf" style="background-color:#caf4bf;">1165<sub>± 169</sub></td>
<td align="right" bgcolor="#dfdfbf" style="background-color:#dfdfbf;">1.66</td>
<td align="right" bgcolor="#d8e6bf" style="background-color:#d8e6bf;">50<sub>± 7</sub></td>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">60.5<sub>± 7.5</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">1022<sub>± 221</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">1.54</td>
</tr>
      <tr>
<th align="left">GPT-5.4-nano</th>
<td align="right" bgcolor="#dce2bf" style="background-color:#dce2bf;">62<sub>± 4</sub></td>
<td align="right" bgcolor="#d7e7bf" style="background-color:#d7e7bf;">75.9<sub>± 4.1</sub></td>
<td align="right" bgcolor="#cef0bf" style="background-color:#cef0bf;">1734<sub>± 276</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">0.22</td>
<td align="right" bgcolor="#efcfbf" style="background-color:#efcfbf;">25<sub>± 6.2</sub></td>
<td align="right" bgcolor="#f4cbbf" style="background-color:#f4cbbf;">32.6<sub>± 7.2</sub></td>
<td align="right" bgcolor="#cbf3bf" style="background-color:#cbf3bf;">1369<sub>± 367</sub></td>
<td align="right" bgcolor="#cbf4bf" style="background-color:#cbf4bf;">0.18</td>
</tr>
      <tr>
<th align="left">GPT-5.4-mini</th>
<td align="right" bgcolor="#dbe3bf" style="background-color:#dbe3bf;">63<sub>± 4</sub></td>
<td align="right" bgcolor="#d5e9bf" style="background-color:#d5e9bf;">77.8<sub>± 4</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">2205<sub>± 389</sub></td>
<td align="right" bgcolor="#d6e8bf" style="background-color:#d6e8bf;">1</td>
<td align="right" bgcolor="#dde2bf" style="background-color:#dde2bf;">45.3<sub>± 6.7</sub></td>
<td align="right" bgcolor="#dae4bf" style="background-color:#dae4bf;">62.8<sub>± 7.5</sub></td>
<td align="right" bgcolor="#d2ecbf" style="background-color:#d2ecbf;">2224<sub>± 847</sub></td>
<td align="right" bgcolor="#d2ecbf" style="background-color:#d2ecbf;">1.05</td>
</tr>
      <tr>
<th align="left">GPT-5.4</th>
<td align="right" bgcolor="#cdf1bf" style="background-color:#cdf1bf;">78.9<sub>± 3.2</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">90.7<sub>± 2.8</sub></td>
<td align="right" bgcolor="#cbf4bf" style="background-color:#cbf4bf;">1299<sub>± 181</sub></td>
<td align="right" bgcolor="#e3dbbf" style="background-color:#e3dbbf;">1.98</td>
<td align="right" bgcolor="#d1edbf" style="background-color:#d1edbf;">58.1<sub>± 6.7</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">74.4<sub>± 6.7</sub></td>
<td align="right" bgcolor="#ccf2bf" style="background-color:#ccf2bf;">1420<sub>± 286</sub></td>
<td align="right" bgcolor="#dee0bf" style="background-color:#dee0bf;">2.29</td>
</tr>
      <tr>
<th align="left">GPT-5.5</th>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">84<sub>± 3.2</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">89.8<sub>± 2.9</sub></td>
<td align="right" bgcolor="#c9f5bf" style="background-color:#c9f5bf;">945<sub>± 110</sub></td>
<td align="right" bgcolor="#f0cfbf" style="background-color:#f0cfbf;">2.89</td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">58.7<sub>± 6.5</sub></td>
<td align="right" bgcolor="#d0eebf" style="background-color:#d0eebf;">74.4<sub>± 6.7</sub></td>
<td align="right" bgcolor="#cdf2bf" style="background-color:#cdf2bf;">1491<sub>± 336</sub></td>
<td align="right" bgcolor="#f5c9bf" style="background-color:#f5c9bf;">4.8</td>
</tr>
    </tbody>
  </table>


</div>
</details>

<br>
<br>


🔃 **Reproducibility:** the evaluation scripts in [`scripts/`](scripts/) allow to fully reproduce our evaluation, follow the instructions below to setup the environment and get started. For size reason, the raw results of all evaluation are not provided, but per-question metrics can be found in [a `.csv` file](data/processed_results_all.csv).

---

### 🗂️ Codebase Structure
-   `conf/`: Configuration files (Hydra).
-  `notebooks/`: Jupyter notebooks for analysis and visualization.
-  **`scripts/`**: Scripts for reproducing evaluations.
-   `src/reasoning_blind_spots/`: Source code package.
    -   `dataset.py`: Dataset loading logic.
    -   `grader.py`: Scorer/Grader/Verifier logic for text outputs.
    -   `solver.py`: Solver/Generator logic for text outputs.
    -   `image_solver.py`: Custom solver for image generation tasks (OpenAI/Google).
    -   `image_grader.py`: Scorer/Grader for image generation outputs.
    -   `task.py`: Inspect AI task definition.
-   [**`main.py`**](main.py): Entry point for running the benchmark with Inspect AI.
-   [`grade_only.py`](grade_only.py): Utility script to perform grading only on provided generations.


### 🛠️ Development
If you want to contribute to the codebase, modify it or simply perform/repeat the evaluation of a model, you can follow the instructions below to set up your development environment and run the evaluation pipeline.

<details>
<summary> Environment Setup </summary>
To set up the environment for this project, follow these steps:

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e ".[dev]"

    # (Optional) if you want to use locally hosted vLLM models
    pip install -e ".[vllm]"
    ```
    (This installs the project in editable mode along with dependencies from `pyproject.toml` and development tools)

3.  **Setup Pre-commit Hooks:**
    Initialize the pre-commit hooks to ensure code quality checks run before every commit:
    ```bash
    pre-commit install
    ```

4.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    OPENAI_API_KEY=...
    GOOGLE_API_KEY=...
    OSS_INFERENCE_OPENAI_API=...
    ```
</details>

<details>
<summary> Running the Benchmark </summary>

To **run the benchmark** using Inspect AI you can execute the [`main.py`](./main.py) script with your chosen configuration. Default configurations are provided in [`config.yaml`](./conf/config.yaml).  Parameters can be overridden via command line arguments. Moreover, [`local_vllm.yaml`](./conf/local_vllm.yaml) provides an *untested* configuration example for local model evaluation.

Inspect AI supports various backends (OpenAI, Anthropic, Google, local models via vLLM, etc.).
For our experiments we will mainly use Gemini/OpenAI models or open-weights models. For the latter we rely on an external service hosting the models with vLLM and exposing an OpenAI API.


Example command to run the benchmark with a specific model:
```bash
# Default config with overrides
python main.py \
    solver.model_name="gemini-2.0-flash-lite" \
    solver.backend="google" \
    solver.generate_config.reasoning_tokens=0

# Or, call the main script using a specific config file
python main.py --config-name local_vllm
```

**Image Generation Tasks:**
The benchmark also supports image generation tasks (text-to-image, image-gen). To run image generation evaluations:
```bash
# Using OpenAI GPT-Image
python main.py --config-name image_gen \
    solver.model_name="gpt-image-1" \
    solver.backend="openai"

# Using Google Imagen
python main.py --config-name image_gen \
    solver.model_name="gemini-2.5-flash-image" \
    solver.backend="google"
```

**Tool-enabled text evaluations:**
Inspect AI can now expose `code_execution()` and `web_search()` to solver models on text-output tasks with a bounded multi-step loop.

For tool-enabled runs, the relevant knobs are:
- `solver.tools.enabled`
- `solver.tools.code_execution`
- `solver.tools.web_search`
- `solver.tools.web_search_providers`
- `solver.tools.max_additional_messages`
- `sandbox` (required for client-side tool execution, e.g. `sandbox: docker`)

For `solver.tools.web_search`, native OpenAI and Gemini backends automatically use their internal web-search tool when no provider override is given. Self-hosted and OpenAI-compatible endpoints such as `openai-api/...`, AI inference services, and local vLLM require an explicit external provider, for example `solver.tools.web_search_providers=["tavily"]`. For that, a TAVILY_API_KEY will also be needed.

The model catalog in [`data/models_info.csv`](data/models_info.csv) includes a `tool_calling` column indicating whether a model can be used with text-mode tools in this benchmark, either natively or through Inspect AI tool emulation.

</details>

<details>
<summary> Inspecting Samples and Results </summary>
Inspect AI provides a built-in dashboard that allows to visualize the results of the benchmark runs, what was generated by both the solver and the grader, and various metrics (e.g. accuracy, token counts, ...)

To **visualize the results**, you can launch Inspect AI visualization tool:
```bash
inspect view --log-dir ./outputs/<your-run-dir>/
```
Inspect AI also provides a VSCode extension that allows to visualize the results directly within VSCode.

</details>
<details>
<summary> Analysis </summary>

Using the notebooks in the `notebooks/` folder, you can analyze the results by loading the `.eval` files generated by Inspect AI runs. For example, you can use the `cost_tracking.ipynb` notebook to analyze the cost of different model runs based on token usage.
</details>
