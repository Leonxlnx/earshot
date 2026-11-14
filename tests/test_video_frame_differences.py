"""Chunked grey-frame arithmetic must match the original vectorized calculation exactly."""
import numpy as np
import pytest
from earshot import video

@pytest.mark.parametrize('frames',[0,1,2,255,256,257,258,511,512,513,1025])
@pytest.mark.parametrize('tail',[0,13])
def test_frame_differences_match_vectorized_reference(frames,tail):
    rng=np.random.default_rng(31+frames)
    raw=rng.integers(0,256,size=frames*video.FRAME_W*video.FRAME_H+tail,dtype=np.uint8).tobytes()
    source=np.frombuffer(raw,np.uint8)
    image=source[:len(source)//(video.FRAME_W*video.FRAME_H)*(video.FRAME_W*video.FRAME_H)].reshape(-1,video.FRAME_H,video.FRAME_W).astype(np.float32)
    expected=np.concatenate([[0.0],np.abs(np.diff(image,axis=0)).mean(axis=(1,2))/255]).astype(np.float32)
    actual=video._frame_differences(raw)
    assert actual.dtype==np.float32
    assert actual.tobytes()==expected.tobytes()

@pytest.mark.parametrize('frames',[1,257,514])
def test_black_white_boundaries(frames):
    data=np.zeros((frames,video.FRAME_H,video.FRAME_W),np.uint8)
    data[1::2]=255
    actual=video._frame_differences(data.tobytes())
    assert actual[0]==0
    assert np.array_equal(actual[1:],np.ones(frames-1,np.float32))
