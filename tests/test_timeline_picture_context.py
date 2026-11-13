"""Picture-only media has known silence; avoid scanning an artificial full-length waveform."""
import subprocess
import wave
import pytest
from earshot import config
from earshot import timeline as tl

@pytest.mark.parametrize('has_audio',[False,True])
def test_known_picture_silence_does_not_scan_full_audio_context(tmp_path,monkeypatch,has_audio):
    monkeypatch.setattr(config,'EMB_CACHE',str(tmp_path/'emb'))
    monkeypatch.setattr(config,'DEC_CACHE',str(tmp_path/'decoded'))
    if has_audio:
        path=str(tmp_path/'silence.wav')
        with wave.open(path,'wb') as out:
            out.setnchannels(2);out.setsampwidth(2);out.setframerate(tl.SR)
            out.writeframes(bytes(6*tl.SR*4))
    else:
        path=str(tmp_path/'picture.mp4')
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=32x32:r=2:d=6',
                        '-an','-c:v','mpeg4','-y',path],check=True,capture_output=True)
    seen={'level':[],'grid':[]}
    original_level,original_grid=tl.level_envs,tl.music_grid
    def level(x):
        seen['level'].append(len(x))
        return original_level(x)
    def grid(x):
        seen['grid'].append(len(x))
        return original_grid(x)
    monkeypatch.setattr(tl,'level_envs',level)
    monkeypatch.setattr(tl,'music_grid',grid)
    result=tl.timeline(path,start=1,end=2,cache=False)
    assert result['duration']==1 and result['context']==[0,6]
    assert result['has_audio']==has_audio
    assert result['music']['note']=='no steady pulse found'
    assert all(v==0 for v in result['model_calls'].values())
    assert seen['level']==([6*tl.SR] if has_audio else [])
    assert seen['grid']==[6*tl.SR if has_audio else 5*tl.SR]
