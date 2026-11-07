import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shadowtrack.probe import sha256
from shadowtrack.sync.api import TimeAlignment
from shadowtrack.sync.audio import correlate, extract_audio, informativeness
from shadowtrack.sync.engine import analyze, build_models
from shadowtrack.sync.fit import fit_intervals, predict_interval
from shadowtrack.sync.local import build_local, save_local, load_local


def anchors(offset=2., slope=1., times=(0.,100.,200.,300.), width=.01):
    return [{"id":str(i),"x":[t-width,t+width],"y":[slope*t+offset-width,slope*t+offset+width]}
            for i,t in enumerate(times)]


def api_for(a,b,events):
    model = fit_intervals(events)
    model.update(id="1:s0->2:s0",node_a="1:s0",node_b="2:s0")
    doc = {"pair_models":[model],"reference_node":"2:s0","reference_support":[model["y_support"]],
           "global_time_basis":"synthetic arbitrary reference"}
    return TimeAlignment(doc,{"1":a,"2":b})


def test_constant_offset_interval_bounds():
    fit = fit_intervals(anchors(times=(1,5)))
    assert fit["status"] == "SUPPORTED"
    assert fit["model"] == "constant_offset"
    assert fit["intercept_seconds"] == pytest.approx(2,abs=1e-6)
    assert fit["drift_ppm"] is None
    lo,hi = predict_interval(fit,3)
    assert lo < 5 < hi


def test_drift_robust_regression():
    events = anchors(offset=.9,slope=1.001,times=(10,100,200,400,600))
    events.append({"id":"bad","x":[350,350.01],"y":[800,800.01]})
    fit = fit_intervals(events)
    assert fit["status"] == "SUPPORTED"
    assert fit["drift_ppm"] == pytest.approx(1000,abs=100)
    assert fit["outlier_ids"] == ["bad"]
    assert fit["rms_center_residual_seconds"] < .03


def test_inconsistent_two_anchors_cannot_be_forced():
    fit = fit_intervals([{"id":"a","x":[0,0],"y":[1,1]},
                         {"id":"b","x":[5,5],"y":[8,8]}])
    assert fit["status"] == "INCONSISTENT"


def test_no_shared_anchors_is_uncalibrated():
    assert fit_intervals([])["status"] == "UNCALIBRATED"
    a = build_local("1",[0,.04,.08],25)
    api = TimeAlignment({"reference_node":"1:s0","reference_support":[],"pair_models":[],"global_time_basis":"test"},{"1":a})
    assert api.map_time("1",.04) == (None,None,"unavailable")


def test_duplicate_anchor_ids_not_independent_evidence():
    events = anchors(times=(1,2))
    events[1]["id"] = events[0]["id"]
    assert fit_intervals(events)["status"] == "INCONSISTENT"


def correction():
    return {"id":"jump","checked":True,"kind":"clock_discontinuity","before_pts":.08,
            "after_pts":10.12,"capture_delta_interval":[.03,.06],"estimated_capture_delta":.04,
            "source":{"manifest":"test"}}


def test_confirmed_piecewise_jump_and_no_gap_interpolation():
    t = build_local("1",[0,.04,.08,10.12,10.16],25,[correction()])
    assert t.estimate == pytest.approx([0,.04,.08,.12,.16])
    assert np.all(np.diff(t.estimate)>0)
    assert len(t.segments) == 2
    assert t.at(5) is None
    assert t.at(10.16)["interval"] == pytest.approx([.15,.18])
    assert t.inverse(.16) == pytest.approx(10.16)


@pytest.mark.parametrize("checked,kind",[(False,"clock_discontinuity"),(True,"real_missing_frames")])
def test_unconfirmed_or_real_missing_frames_not_compressed(checked,kind):
    c = dict(correction(),checked=checked,kind=kind)
    t = build_local("1",[0,.04,.08,10.12,10.16],25,[c])
    assert t.estimate[-1] == pytest.approx(10.16)
    assert t.at(5) is None
    assert t.anomalies[0]["corrected"] is False


def test_missing_timestamp_not_interpolated_as_capture():
    t = build_local("1",[0,.04,None,.12,.16],25)
    assert np.all(np.diff(t.estimate)>0)
    assert t.frame_time(2)["status"] == "INDEX_ONLY_UNSUPPORTED"
    assert t.frame_time(2)["raw_media_pts"] is None
    assert t.at(.08) is None


def test_duplicate_raw_pts_unavailable_but_frame_index_is_monotone():
    t = build_local("1",[0,.04,.04,.08,.12],25)
    assert t.at(.04) is None
    assert np.all(np.diff(t.estimate)>0)


def test_backwards_raw_pts_no_ambiguous_raw_mapping():
    t = build_local("1",[0,.04,.08,.01,.05,.09],25)
    assert t.at(.02) is None  # Overlapping raw domains cannot identify a frame.
    assert np.all(np.diff(t.estimate)>0)


def test_nonadjacent_correction_fails_closed():
    c = dict(correction(),before_pts=.04)
    with pytest.raises(ValueError,match="non-adjacent"):
        build_local("1",[0,.04,.08,10.12],25,[c])


def test_fps_mismatch_uses_pts_not_common_frame_index():
    a = build_local("1",np.arange(0,6,.04).tolist(),25)
    b = build_local("2",np.arange(0,8,.05).tolist(),20)
    api = api_for(a,b,anchors(offset=1,times=(1,4)))
    mapped,unc,status = api.map_time("1",2)
    assert mapped == pytest.approx(3,abs=1e-6)
    assert a.frame_time(50)["relative_capture_time_estimate"] == pytest.approx(2)
    assert b.frame_time(50)["relative_capture_time_estimate"] == pytest.approx(2.5)
    assert status == "SUPPORTED_CONDITIONAL"


def test_mapping_inverse_and_no_extrapolation():
    a = build_local("1",np.arange(0,8,.04).tolist(),25)
    b = build_local("2",np.arange(0,10,.05).tolist(),20)
    api = api_for(a,b,anchors(times=(1,5)))
    v,unc,status = api.map_time("1",3)
    inverse,raw_unc,_ = api.inverse_map_time("1",v)
    assert inverse == pytest.approx(3,abs=1e-6)
    assert unc["lower_seconds"] < v < unc["upper_seconds"]
    assert raw_unc["raw_pts_interval"][0] < inverse < raw_unc["raw_pts_interval"][1]
    assert api.map_time("1",7)[2] == "unavailable"
    assert api.map_time("2",8)[2] == "unavailable"
    assert api.map_time("unknown",2)[2] == "unavailable"


def test_evidence_uncertainty_cannot_disappear():
    fit = fit_intervals(anchors(width=.25,times=(1,5)),systematic_seconds=.02)
    lo,hi = predict_interval(fit,3)
    assert hi-lo >= 1.


def test_serialized_local_mapping_roundtrip(tmp_path):
    t = build_local("1",[0,.04,.08,10.12,10.16],25,[correction()])
    path = tmp_path/"local.npz"
    save_local(t,path)
    loaded = load_local({"camera_id":"1","local_segments":t.segments,"anomalies":t.anomalies,"nominal_fps":25},path)
    assert loaded.at(10.16) == t.at(10.16)


def test_unchecked_event_not_promoted(tmp_path):
    manifest = tmp_path/"manifest.json"
    manifest.write_text("[]")
    events = [{"id":"candidate","checked":False,"observations":{"1":[1,2],"2":[1,2]}}]
    timelines = {cid:build_local(cid,[0,1,2,3],1) for cid in ("1","2")}
    models,considered = build_models(timelines,events,tmp_path,{})
    assert not models
    assert considered[0]["exclusion_reason"] == "unchecked"


def test_checked_anchor_crossing_real_gap_rejected(tmp_path):
    manifest = tmp_path/"manifest.json"
    manifest.write_text("[]")
    events = [{"id":"crossgap","checked":True,"reviewer":"test","description":"event",
               "kind":"physical","observations":{"1":[.04,10.12],"2":[.04,.08]},
               "source":{"manifest":"manifest.json","sha256":sha256(manifest)}}]
    timelines = {"1":build_local("1",[0,.04,.08,10.12,10.16],25),
                 "2":build_local("2",[0,.04,.08,.12],25)}
    models,considered = build_models(timelines,events,tmp_path,{})
    assert not models
    assert "break" in considered[0]["exclusion_reason"]


def test_silence_ccf_uninformative_not_negative_sync():
    x = np.full(80000,8,dtype=np.int16)
    result = correlate(x,x,8000)
    assert result["status"] == "UNINFORMATIVE"
    assert result["lag_seconds"] is None


def test_audio_ccf_known_nonperiodic_delay():
    rng = np.random.default_rng(3)
    original = rng.normal(0,2000,8000*10).astype(np.int16)
    delay = 1200
    shifted = np.r_[np.zeros(delay,dtype=np.int16),original[:-delay]]
    result = correlate(shifted,original,8000,.5)
    assert result["status"] == "CANDIDATE"
    assert result["lag_seconds"] == pytest.approx(.15,abs=.001)
    assert result["checked"] is False


def test_periodic_audio_peak_ambiguous():
    t = np.arange(80000)/8000
    x = (5000*np.sin(2*np.pi*200*t)).astype(np.int16)
    result = correlate(x,x,8000,.5)
    assert result["status"] == "AMBIGUOUS"


@pytest.mark.integration
def test_real_ffmpeg_audio_decode_and_audio_pts(tmp_path):
    ffmpeg,ffprobe = Path(".tools/ffmpeg.exe").resolve(),Path(".tools/ffprobe.exe").resolve()
    if not ffmpeg.is_file() or not ffprobe.is_file():
        pytest.skip("FFmpeg integration tools unavailable")
    import os
    old = {name:os.environ.get(name) for name in ("SHADOWTRACK_FFMPEG","SHADOWTRACK_FFPROBE")}
    try:
        os.environ["SHADOWTRACK_FFMPEG"],os.environ["SHADOWTRACK_FFPROBE"] = str(ffmpeg),str(ffprobe)
        video = tmp_path/"audio.mkv"
        subprocess.run([str(ffmpeg),"-v","error","-f","lavfi","-i","sine=frequency=731:sample_rate=8000:duration=2",
                        "-c:a","pcm_alaw",str(video)],check=True,capture_output=True,timeout=30)
        camera = {"camera_id":"test","file":str(video),"sha256":sha256(video),
                  "audio_streams":[{"sample_rate":"8000","channels":1,"codec_name":"pcm_alaw"}]}
        record,samples = extract_audio(camera,tmp_path/"out")
        assert len(samples) == 16000
        assert record["metrics"]["informative"]
        assert record["audio_video_latency"] == "UNVERIFIED"
        assert record["commands"][1][record["commands"][1].index("-select_streams")+1] == "a:0"
    finally:
        for name,value in old.items():
            if value is None:
                os.environ.pop(name,None)
            else:
                os.environ[name] = value


def test_analyze_reuses_saved_pts_and_api_can_load(tmp_path):
    # Integration of evidence loader, persisted local maps, regression and API;
    # any accidental full probe would fail on these deliberately non-media inputs.
    root = tmp_path
    audit = root/"audit"
    audit.mkdir()
    cameras,hashes,frame_hashes,files = [],{},{},{}
    for cid,fps,offset in [("1",25,5),("2",20,0)]:
        video = root/f"c{cid}.mkv"
        video.write_bytes(b"saved probe fixture, not a decodable video")
        folder = audit/cid/"timestamps"
        folder.mkdir(parents=True)
        frames = [{"best_effort_timestamp_time":str(v)} for v in np.arange(offset,offset+4,1/fps)]
        p = folder/"frames.json"
        p.write_text(json.dumps({"frames":frames}))
        digest = sha256(video)
        cameras.append({"file":str(video),"key":cid,"sha256":digest,"nominal_fps":fps})
        files[cid],hashes[video.name],frame_hashes[cid] = video.name,digest,sha256(p)
    stats_path = audit/"stats.json"
    stats_path.write_text(json.dumps({"cameras":cameras}))
    source = root/"event_manifest.json"
    source.write_text("[]")
    events = [{"id":f"e{i}","kind":"synthetic","checked":True,"reviewer":"test","description":"physical flash",
               "observations":{"1":[v+5-.01,v+5+.01],"2":[v-.01,v+.01]},
               "source":{"manifest":"event_manifest.json","sha256":sha256(source)}} for i,v in enumerate([.5,3.])]
    evidence = root/"evidence.json"
    evidence.write_text(json.dumps({"camera_files":files,"input_sha256":hashes,"frame_pts_sha256":frame_hashes,"events":events}))
    before = sha256(stats_path)
    result = analyze(stats_path,evidence,root/"output",reference_camera="2",audio=False,root=root)
    assert result["full_video_probe_repeated"] is False
    assert sha256(stats_path) == before
    api = TimeAlignment.load(root/"output/time_alignment.json")
    assert api.map_time("1",6.5)[0] == pytest.approx(1.5,abs=.001)
    assert api.map_time("1",8.9)[2] == "unavailable"


def test_inconsistent_graph_routes_fail_closed():
    timelines = {cid:build_local(cid,np.arange(0,121,.04).tolist(),25) for cid in ("1","2","3")}
    models = []
    for a,b,offset in [("1","2",1),("2","3",1),("1","3",10)]:
        model = fit_intervals(anchors(offset=offset,times=(1,100)))
        model.update(id=a+b,node_a=a+":s0",node_b=b+":s0")
        models.append(model)
    api = TimeAlignment({"pair_models":models,"reference_node":"3:s0","reference_support":[[0,110]],"global_time_basis":"test"},timelines)
    assert api.map_time("1",30) == (None,None,"inconsistent")


@pytest.mark.integration
def test_real_ffmpeg_clock_jump_and_real_missing_frames(tmp_path):
    from shadowtrack.probe import probe_video
    ffmpeg,ffprobe = Path(".tools/ffmpeg.exe").resolve(),Path(".tools/ffprobe.exe").resolve()
    if not ffmpeg.is_file() or not ffprobe.is_file():
        pytest.skip("FFmpeg integration tools unavailable")
    timelines = []
    for name,filtergraph in [("clock","setpts=PTS+if(gte(N\\,12)\\,10/TB\\,0)"),
                              ("missing","select=not(between(n\\,12\\,15))")]:
        video = tmp_path/(name+".mkv")
        subprocess.run([str(ffmpeg),"-v","error","-f","lavfi","-i","testsrc2=size=160x120:rate=25:duration=1",
                        "-vf",filtergraph,"-fps_mode","passthrough","-c:v","libx264","-bf","0",str(video)],
                       check=True,capture_output=True,timeout=30)
        folder = tmp_path/name
        probe_video(video,folder,str(ffprobe),30)
        pts = [float(f["best_effort_timestamp_time"]) for f in json.loads((folder/"frames.json").read_text())["frames"]]
        if name == "clock":
            assert len(pts) == 25
            c = {"id":"constructed_clock_jump","kind":"clock_discontinuity","checked":True,
                 "before_pts":.44,"after_pts":10.48,"capture_delta_interval":[.04,.04],
                 "estimated_capture_delta":.04,"source":{"filtergraph":filtergraph}}
            local = build_local(name,pts,25,[c])
            assert local.estimate[-1] == pytest.approx(.96)
        else:
            assert len(pts) == 21
            local = build_local(name,pts,25)
            assert local.estimate[-1] == pytest.approx(.96)
            assert local.at(.50) is None
            assert local.anomalies[0]["raw_delta_seconds"] == pytest.approx(.2)
        timelines.append(local)
    assert all(np.all(np.diff(t.estimate)>0) for t in timelines)


def test_missing_alignment_file_returns_unavailable(tmp_path):
    from shadowtrack.sync import map_time
    assert map_time("1",2,tmp_path/"not_created.json") == (None,None,"unavailable")


def test_piecewise_offsets_keep_gap_and_monotone_global_time():
    a = build_local("1",[0,.04,.08,10.12,10.16,10.20],25)
    b = build_local("2",np.arange(0,20,.04).tolist(),25)
    models = []
    for seg,times,offset in [("s0",(0,.08),2),("s1",(10.12,10.20),3)]:
        m = fit_intervals(anchors(offset=offset,times=times,width=0))
        m.update(id=seg,node_a="1:"+seg,node_b="2:s0")
        models.append(m)
    api = TimeAlignment({"pair_models":models,"reference_node":"2:s0","reference_support":[m["y_support"] for m in models],"global_time_basis":"synthetic"},{"1":a,"2":b})
    times = [api.map_time("1",p)[0] for p in [0,.04,.08,10.12,10.16,10.20]]
    assert np.all(np.diff(times)>0)
    assert times[1] == pytest.approx(2.04)
    assert times[4] == pytest.approx(13.16)
    assert api.map_time("1",5) == (None,None,"unavailable")
