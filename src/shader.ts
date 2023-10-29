import {
  ComposableShader,
  SequenceShader,
  ShaderAndSize,
  applyTemplate,
  bufferF32,
  gpuTiming,
  mapN,
} from "thimbleberry";
import wgsl from "./reduce_simple.wgsl?raw";

const workSize = 2 ** 8;

/**
 * Return a reduction shader sequence, intialized to run on benchmarking data.
 *
 * The example shader is built for a fixed size of input data.
 * For a typically sized benchmark, three different sized versions of 
 * the reduction shader will be created, each reducing the output of the previous.
 * Dispatches for the three shaders are queued to the
 * GPU in a single command buffer. The three shaders run in sequence
 * on the GPU, resulting in a buffer containing a single value.
 */
export function simpleShader(
  device: GPUDevice,
  inputSize: number // size in elements
): ShaderAndSize {
  const srcData = mapN(inputSize, () => Math.random());
  const source = bufferF32(device, srcData);
  const buffers = [source, ...reduceBuffers(device, inputSize)];

  const shaders = grouped(buffers, 2, 1)
    .slice(0, -1)
    .map(([src, dest], i) =>
      reduceShader(device, src, dest, inputSize / workSize ** i)
    );

  const shader = new SequenceShader(shaders, "reduce_simple");
  return { shader, srcSize: inputSize * Float32Array.BYTES_PER_ELEMENT };
}

/** buffers for reduction results */
function reduceBuffers(device: GPUDevice, inputSize: number): GPUBuffer[] {
  return [...reduceBufferSizes(inputSize)].map((size) => {
    return device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
  });
}

/** sizes of buffers for reduction results, in bytes */
function* reduceBufferSizes(inputSize: number): Generator<number> {
  const elemSize = Float32Array.BYTES_PER_ELEMENT;
  let size = inputSize * elemSize;
  do {
    size = Math.max(size / workSize, elemSize);
    yield size;
  } while (size > elemSize);
}

/** return a reduce shader for a fixed input size */
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
