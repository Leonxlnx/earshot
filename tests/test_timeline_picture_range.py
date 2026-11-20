"""Picture-only timeline ranges must not silently become empty or wrap from the end."""
import subprocess
import pytest
from earshot import config
from earshot.timeline import timeline

@pytest.fixture
def picture(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'EMB_CACHE',str(tmp_path/'cache'))
    path=str(tmp_path/'silent.mp4')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=25:d=2',
                    '-c:v','mpeg4','-y',path],check=True,capture_output=True)
    return path

@pytest.mark.parametrize('bounds',[{'end':-1},{'start':3},{'start':1.5,'end':.5},
                                   {'start':1,'end':1},{'start':2},
                                   {'start':1,'end':1.000001},{'end':0.000001}])
def test_invalid_picture_range_never_creates_cached_success(picture,bounds):
    for _ in range(2):
        with pytest.raises(ValueError,match='nothing to analyse'):
            timeline(picture,**bounds)

@pytest.mark.parametrize('bounds,expected',[({'start':.5,'end':1.5},(.5,1.5)),
                                           ({'start':1,'end':1.05},(1,1.05)),
                                           ({'start':-1,'end':1},(0,1)),
                                           ({'start':1,'end':3},(1,2))])
def test_valid_picture_range_and_cached_repeat_unchanged(picture,bounds,expected):
    first=timeline(picture,**bounds)
    second=timeline(picture,**bounds)
    assert (first['start'],first['end'])==expected
    assert first['duration']==pytest.approx(expected[1]-expected[0])
    assert not first['cached'] and second['cached']
    assert (second['start'],second['end'],second['duration'])==(first['start'],first['end'],first['duration'])
    assert not first['has_audio']
    assert all(count==0 for count in first['model_calls'].values())
