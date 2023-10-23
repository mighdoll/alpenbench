import {
  ComposableShader,
  applyTemplate,
  bufferF32,
  gpuTiming,
  initGpuTiming,
  mapN,
  printBuffer,
} from "thimbleberry";

// these files (from stoneberry) should get cleaned up and graduate to thimbleberry. 
import { benchDevice } from "./lib/benchDevice.js";
import { logCsvReport } from "./lib/benchReport.js";
import { benchShader } from "./lib/benchShader.js";

import wgsl from "./reduce_simple.wgsl?raw";

/*
  This shows using some of the thimbleberry utilities to benchmark a shader.
  Basically, to benchmark a shader, you need to:
    0 call initGpuTiming() on a GPUDevice with 'timestamp-query' enabled.
    1 expose the commands() api from your shader.
    2 use gpuTiming?.timestampWrites in the shader's beginComputePass()
  Then you can call:
    3 benchShader() and logCsvReport() to benchmark the shader and report the results.
  
  You'll note that the stoneberry shaders are written in a different style: 
  with classes, reactivity, resource mgmt, testing support, etc. Stoneberry's aiming
  for more reusability, but this is simpler. 
*/

main();

const inputSize = 2 ** 23;
const workSize = 2 ** 8;

async function main(): Promise<void> {
  const time = Date.now().toString();
  const device = await benchDevice();

  // initializes and sets up gpuTiming as a convenient global.
  // (I don't like globals btw. Maybe something like alpenglow's DeviceContext would be a better api..)
  initGpuTiming(device);

  const srcData = mapN(inputSize, () => Math.random());
  const source = bufferF32(device, srcData);
  const buffers = [source, ...reduceBuffers(device)];

  const shaders = grouped(buffers, 2, 1)
    .slice(0, -1)
    .map(([src, dest], i) =>
      reduceShader(device, src, dest, inputSize / workSize ** i)
    );

  // (internally benchShader uses a ShaderGroup which submits the commands 
  //  from all the shaders to the device queue in one command buffer)
  const { averageClockTime, fastest } = await benchShader(
    { device, runs: 100 },
    ...shaders
  );

  // I use printBuffer and printTexture constantly during development!
  await printBuffer(device, buffers.slice(-1)[0]);
  const expected = srcData.reduce((a, b) => a + b);
  console.log({ expected });

  // logging to csv format as you saw in stoneberry.
  // I use it during development to quick check performance ideas
  // and keep the csv file around to track performance regressions
  // and occasionally import into a chart tool like tableau or google sheets.
  logCsvReport([fastest], averageClockTime, inputSize / 4, "reduce_s", time);
}

/** buffers for reduction results */
function reduceBuffers(device: GPUDevice): GPUBuffer[] {
  return [...reduceBufferSizes()].map((size) => {
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  });
}

/** sizes of buffers for reduction results, in bytes */
function* reduceBufferSizes(): Generator<number> {
  const elemSize = Float32Array.BYTES_PER_ELEMENT;
  let size = inputSize * elemSize;
  do {
    size = Math.max(size / workSize, elemSize);
    yield size;
  } while (size > elemSize);
}

// return a reduce shader for a fixed input size
// [btw I'm inspired to look further at some of your abstractions 
//  that make std bind groups and pipelines]
function reduceShader(
  device: GPUDevice,
  src: GPUBuffer,
  dest: GPUBuffer,
  inputSize: number,
  label: string = "reduce"
): ComposableShader {
  const dispatches = Math.ceil(inputSize / workSize);
  const pipeline = reducePipeline(device, inputSize);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dest } },
    ],
  });

  // support for the ComponsableShader api
  function commands(encoder: GPUCommandEncoder): void {
    // I add these routinely to get timing per shader.
    // (ShaderGroup groups these timings, so later we can find the fastest run)
    const timestampWrites = gpuTiming?.timestampWrites(
      `${label} ${dispatches}`
    );
    const pass = encoder.beginComputePass({ timestampWrites });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatches);
    pass.end();
  }

  return { commands };
}

/** pipeline compiled for a fixed inputSize */
function reducePipeline(
  device: GPUDevice,
  inputSize: number
): GPUComputePipeline {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0, // input
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1, // output
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const module = device.createShaderModule({
    // Thimbleberry has a lightweight template scheme that lives in wgsl comments 
    // to maintain compatibility with tools like wgsl-analyzer. 
    // ... it's ok as an interim thing, but I really would like to see a template
    // facility built into wgsl analyzer or similar VSCode tooling.
    code: applyTemplate(wgsl, { inputSize }), 
  });

  const pipeline = device.createComputePipeline({
    compute: { module, entryPoint: "main" },
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
  });

  return pipeline;
}

/** return an array partitioned into possibly overlapping groups */
function grouped<T>(a: T[], size: number, stride = size): T[][] {
  const groups = [];
  for (let i = 0; i < a.length; i += stride) {
    groups.push(a.slice(i, i + size));
  }
  return groups;
}