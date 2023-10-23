@group(0) @binding(0) var<storage> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> scratch: array<f32, 256>;

const INPUT_SIZE_U = 1024u; //! 1024=inputSize

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    var value = select(0f, input[ global_id.x ], global_id.x < INPUT_SIZE_U);
    scratch[ local_id.x ] = value;

    workgroupBarrier();
    if local_id.x < 255u {
        value = value + scratch[ local_id.x + 1u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 254u {
        value = value + scratch[ local_id.x + 2u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 252u {
        value = value + scratch[ local_id.x + 4u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 248u {
        value = value + scratch[ local_id.x + 8u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 240u {
        value = value + scratch[ local_id.x + 16u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 224u {
        value = value + scratch[ local_id.x + 32u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 192u {
        value = value + scratch[ local_id.x + 64u ];
    }
    workgroupBarrier();
    scratch[ local_id.x ] = value;
    workgroupBarrier();
    if local_id.x < 128u {
        value = value + scratch[ local_id.x + 128u ];
    }
    if local_id.x == 0u {
        output[ workgroup_id.x ] = value;
    }
}