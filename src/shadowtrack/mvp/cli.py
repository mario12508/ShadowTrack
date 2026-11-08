"""Explicit preparation only: starting the web app never replays computer vision."""
from pathlib import Path
import json
import zipfile
from ..detection.common import atomic_bytes,read_json
from .config import Settings


def add_parser(sub):
    p=sub.add_parser('mvp',help='Prepare and serve the CPU ShadowTrack application')
    commands=p.add_subparsers(dest='mvp_command',required=True)
    prepare=commands.add_parser('prepare',help='Import FULL, replay ByteTrack and index SQLite; no YOLO')
    prepare.add_argument('--source',type=Path,required=True)
    prepare.add_argument('--render',action='store_true',help='Also render all camera videos on CPU')
    serve=commands.add_parser('serve');serve.add_argument('--host',default='127.0.0.1');serve.add_argument('--port',type=int,default=8000)
    render=commands.add_parser('render');render.add_argument('--width',type=int,default=960,choices=(960,1280,1920));render.add_argument('--workers',type=int,default=2,choices=(1,2,3))
    commands.add_parser('browser-media',help='Make browser H264 sources, keeping raw PTS discontinuities')
    commands.add_parser('highlights',help='Make short presentation clips from existing annotated videos')
    reid=commands.add_parser('reid');reid.add_argument('--weights',type=Path,required=True);reid.add_argument('--device',default='cpu');reid.add_argument('--samples',type=int,default=4)
    crops=commands.add_parser('reid-crops');crops.add_argument('--samples',type=int,default=4)
    embeddings=commands.add_parser('embeddings-import');embeddings.add_argument('--source',type=Path,required=True)
    commands.add_parser('associate')
    commands.add_parser('publish',help='Publish a closed SQLite snapshot for the native Docker serving database')
    commands.add_parser('appearance-normalize',help='Explicit CPU scene-balanced crop features; no YOLO or source rewriting')
    commands.add_parser('appearance',help='Explicit one-time CPU temporal OSNet cache; normal MTMC recompute does not run it')
    commands.add_parser('mtmc',help='Experimental global assignment, existing OSNet only; no inference')
    review=commands.add_parser('mtmc-preview',help='Private revision-bound candidate clips; not proof of identity')
    review.add_argument('--episode',required=True);review.add_argument('--seconds',type=float,default=6)


def execute(args):
    settings=Settings.environment()
    if args.mvp_command=='serve':
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(settings),host=args.host,port=args.port,log_level='info')
        return 0
    if args.mvp_command=='prepare':
        from ..detection.transport import import_run
        from ..detection.tracking import local_track
        from .prepare import prepare_database
        source=args.source.resolve()
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                if 'export_manifest.json' not in archive.namelist():
                    nested=[n for n in archive.namelist() if n.endswith('/shadowtrack_detection_full.zip')]
                    if len(nested)!=1:raise ValueError('Expected one FULL data export inside outer download')
                    target=settings.artifacts/'input_cache/shadowtrack_detection_full.zip'
                    atomic_bytes(target,archive.read(nested[0]));source=target
        detection=import_run(source,settings.evidence,settings.root/'artifacts/detections',raw_dir=settings.raw)
        config=settings.artifacts/'local_track.toml'
        paths={'detection_run':str(detection),'evidence_dir':str(settings.evidence),'raw_dir':str(settings.raw),'output_dir':str(settings.artifacts/'local_tracks')}
        config.parent.mkdir(parents=True,exist_ok=True)
        text='[paths]\n'+'\n'.join(f'{k}={json.dumps(v.replace(chr(92),"/"))}' for k,v in paths.items())
        text+='\n[tracking]\nconfiguration_name="small_640"\n[bytetrack]\ntrack_activation_threshold=0.30\nlost_track_buffer=45\nminimum_matching_threshold=0.80\nminimum_consecutive_frames=3\n'
        config.write_text(text,encoding='utf-8')
        if not (settings.artifacts/'local_tracks/manifest.json').exists():local_track(config)
        if not settings.database.exists() or not (settings.artifacts/'prepare_manifest.json').exists():prepare_database(settings,detection,settings.artifacts/'local_tracks')
        from .render import browser_sources
        browser_sources(settings)
        if args.render:
            from .render import render_all
            render_all(settings)
            from .highlights import make_highlights
            make_highlights(settings)
        print('MVP prepared; start backend with shadowtrack mvp serve')
    elif args.mvp_command=='render':
        from .render import render_all
        print(render_all(settings,args.width,args.workers)['status'])
    elif args.mvp_command=='browser-media':
        from .render import browser_sources
        browser_sources(settings)
    elif args.mvp_command=='highlights':
        from .highlights import make_highlights
        make_highlights(settings)
    elif args.mvp_command=='reid-crops':
        from .reid import make_crop_package
        print(make_crop_package(settings,args.samples))
    elif args.mvp_command=='embeddings-import':
        from .reid import import_embeddings
        source=args.source
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                names=[n for n in archive.namelist() if n.endswith('embedding_manifest.json')]
                if len(names)!=1:raise ValueError('Expected one embedding manifest')
                prefix=names[0][:-len('embedding_manifest.json')]
                target=settings.artifacts/'reid/imported_embeddings'
                for name in ('embedding_manifest.json','embeddings.npz'):atomic_bytes(target/name,archive.read(prefix+name))
                source=target
        print(import_embeddings(settings,source))
    elif args.mvp_command=='reid':
        from .reid import extract_embeddings,import_embeddings
        path=extract_embeddings(settings,args.weights,args.device,args.samples)
        print(import_embeddings(settings,path))
    elif args.mvp_command=='publish':
        from .snapshot import publish_snapshot
        print(publish_snapshot(settings))
    elif args.mvp_command=='appearance-normalize':
        from .normalized_appearance import prepare as normalize
        print(normalize(settings))
    elif args.mvp_command=='appearance':
        from .appearance import prepare_temporal_features
        print(prepare_temporal_features(settings))
    elif args.mvp_command=='mtmc':
        from .mtmc import run_mtmc
        print(run_mtmc(settings))
    elif args.mvp_command=='mtmc-preview':
        from .mtmc_render import review_highlight
        print(review_highlight(settings,args.episode,args.seconds)['episode'])
    else:
        from .mtmc import run_mtmc
        print(run_mtmc(settings))
    return 0
