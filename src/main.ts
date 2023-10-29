import { benchRunner } from "thimbleberry";
import { simpleShader } from "./shader.js";

/*
  This shows using some of the thimbleberry utilities to benchmark a shader.

  Basically, to benchmark a shader, you need to:
    1) use gpuTiming?.timestampWrites in your shader's beginComputePass()
    2) expose the commands() api from your shader, so the benchmark runner can run it.

  Then you can call:
    3) benchRunner() with a function that sets up your shader and data buffers for benchmarking.
*/

main();

const inputSize = 2 ** 23;

async function main(): Promise<void> {
  await benchRunner([
    { makeShader: (d: GPUDevice) => simpleShader(d, inputSize) },
  ]);
}

// Notes:
// * you can modify the report type, number of warmup runs, etc. by passing
//   additional parameters to benchRunner(). 
// 
// * You can modify the same parameters via url query parameters, 
//   e.g. http://localhost:5173/?reportType=details&precision=4&runs=10
//  
// * The current default report type is "median" (no longer "fastest")
// 
// * There's an example tableau dashboard for stoneberry, ingesting the 
//   csv output from report type "details"
// 
