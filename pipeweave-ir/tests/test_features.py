from pipeweave_ir.features import (
    extract_mlir_features,
    extract_ptx_features,
    extract_sass_features,
    flatten_numeric,
)


def test_extract_mlir_features() -> None:
    text = """
    %0 = tt.load %arg0 : tensor<16xf32>
    %1 = ttg.convert_layout %0 : tensor<16xf32, #blocked>
    %2 = tt.dot %1, %1, %1 : tensor<16x16xf32>
    tt.store %arg1, %2 : tensor<16x16xf32>
    scf.for %i = %c0 to %c4 step %c1 { }
    """
    features = extract_mlir_features(text)
    assert features["operations"]["tt.load"] == 1
    assert features["categories"]["dot"] == 1
    assert features["categories"]["layout_convert"] == 1
    assert features["categories"]["loop"] == 1
    assert features["layouts"]["#blocked"] == 1


def test_extract_ptx_features() -> None:
    text = """
    .version 9.0
    entry:
      ld.global.v4.u32 {%r1,%r2,%r3,%r4}, [%rd1];
      mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f0}, {%r1}, {%r2}, {%f0};
      @%p1 bra DONE;
      atom.global.add.f32 %f1, [%rd2], %f0;
      barrier.sync 0;
    DONE:
      ret;
    """
    features = extract_ptx_features(text)
    assert features["categories"]["load"] == 1
    assert features["categories"]["tensor"] == 1
    assert features["categories"]["atomic"] == 1
    assert features["categories"]["barrier"] == 1
    assert features["categories"]["control"] == 2


def test_extract_sass_features() -> None:
    text = """
        /*0000*/                   LDG.E.128 R4, [R2.64] ;
        /*0010*/              @P0  LDGSTS.E.BYPASS.128 [R8], [R4.64] ;
        /*0020*/                   HMMA.16816.F32 R16, R8, R12, R16 ;
        /*0030*/                   BAR.SYNC 0x0 ;
        /*0040*/                   STG.E.128 [R2.64], R4 ;
    """
    features = extract_sass_features(text)
    assert features["instruction_count"] == 5
    assert features["categories"]["global_load"] == 1
    assert features["categories"]["async_copy"] == 1
    assert features["categories"]["tensor"] == 1
    assert features["categories"]["barrier"] == 1
    assert features["categories"]["global_store"] == 1


def test_flatten_numeric_preserves_sequence_positions() -> None:
    output = {}
    flatten_numeric("launch.grid", [4, 8], output)
    assert output == {"launch.grid.0": 4.0, "launch.grid.1": 8.0}
