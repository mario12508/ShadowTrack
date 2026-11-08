import {test,expect} from '@playwright/test';

for(const mode of ['relative','global','estimated'])test(`legacy ${mode} URL uses native playback without clock switching`,async({page})=>{
 await page.setViewportSize({width:1440,height:2600});
 await page.goto(`/monitor?camera=3&time=129&clock=${mode}&gtime=129`);
 await page.getByRole('button',{name:'Камеры',exact:true}).click();
 const videos=page.locator('.camera-grid video');await expect(videos).toHaveCount(5);
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>v.readyState>=2&&Math.abs(v.currentTime-129)<.2)),{timeout:30000}).toBeTruthy();
 await expect(page.locator('.clock-status,.estimated-button,.clock-button')).toHaveCount(0);
 const before=await videos.evaluateAll((vs:HTMLVideoElement[])=>vs.map(v=>v.currentTime));
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[],times:number[])=>vs.every((v,i)=>!v.paused&&v.currentTime>times[i]+.7),before),{timeout:20000}).toBeTruthy();
 await page.getByRole('button',{name:'Пауза',exact:true}).click();
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>v.paused))).toBeTruthy();
 await page.getByRole('slider',{name:'Временная шкала'}).fill('410');
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>Math.abs(v.currentTime-410)<.2)),{timeout:20000}).toBeTruthy();
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 const later=await videos.evaluateAll((vs:HTMLVideoElement[])=>vs.map(v=>v.currentTime));
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[],times:number[])=>vs.every((v,i)=>v.currentTime>times[i]+.5),later),{timeout:20000}).toBeTruthy();
 await page.screenshot({path:`../private_outputs/mvp_browser_smoke/playback-${mode}.png`,fullPage:true});
});

test('five players do not seek or move backwards while overlays and slow map requests update',async({page})=>{
 await page.setViewportSize({width:1440,height:2600});
 await page.route('**/api/overlays/**',async route=>{await new Promise(r=>setTimeout(r,500));await route.continue()});
 await page.goto('/monitor?camera=1&time=10&clock=estimated&gtime=10');
 await page.getByRole('button',{name:'Камеры',exact:true}).click();
 const videos=page.locator('.camera-grid video');await expect(videos).toHaveCount(5);
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>v.readyState>=2&&Math.abs(v.currentTime-10)<.2)),{timeout:30000}).toBeTruthy();
 await videos.evaluateAll((vs:HTMLVideoElement[])=>vs.forEach(v=>{const s=(v as any)._proof={seeks:0,backwards:0,previous:v.currentTime,start:v.currentTime,frames:v.getVideoPlaybackQuality().totalVideoFrames,drops:v.getVideoPlaybackQuality().droppedVideoFrames,callbacks:0};const shown=()=>{s.callbacks++;v.requestVideoFrameCallback(shown)};v.requestVideoFrameCallback(shown);v.addEventListener('seeking',()=>s.seeks++);v.addEventListener('timeupdate',()=>{if(v.currentTime<s.previous-.03)s.backwards++;s.previous=v.currentTime})}));
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 await expect.poll(()=>videos.evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>v.currentTime>24)),{timeout:25000}).toBeTruthy();
 const proof=await videos.evaluateAll((vs:HTMLVideoElement[])=>vs.map(v=>({...((v as any)._proof),end:v.currentTime,total:v.getVideoPlaybackQuality().totalVideoFrames-((v as any)._proof.frames),dropped:v.getVideoPlaybackQuality().droppedVideoFrames-((v as any)._proof.drops)})));
 await expect(page.locator('.overlay-error')).toHaveCount(0);
 expect(proof.every(p=>p.seeks===0&&p.backwards===0&&p.callbacks>=180&&p.dropped<Math.max(4,p.total*.15))).toBeTruthy();
 await test.info().attach('native-playback-proof',{body:JSON.stringify(proof),contentType:'application/json'});
});

test('first foyer exit has the same automatic ID on both views and one moving map dot',async({page,request})=>{
 const overlays=await Promise.all(['1','2'].map(async cam=>(await request.get(`/api/overlays/${cam}?start=.5&end=1.5`)).json()));
 const people=overlays.map(d=>d.frames.flatMap((f:any)=>f.tracks).filter((t:any)=>t.display_id.endsWith('-T001')));
 expect(people.every(p=>p.length>5)).toBeTruthy();
 const ids=people.map(p=>new Set(p.map((t:any)=>t.global_episode_id)));expect([...ids[0]]).toEqual([...ids[1]]);expect(ids[0].size).toBe(1);
 const id=[...ids[0]][0],a=await (await request.get('/api/floor?time=.8')).json(),b=await (await request.get('/api/floor?time=2')).json();
 const p=a.markers.filter((m:any)=>m.global_episode_id===id),q=b.markers.filter((m:any)=>m.global_episode_id===id);
 expect(p).toHaveLength(1);expect(q).toHaveLength(1);expect(p[0].camera_sources).toEqual(['1','2']);
 expect(q[0].position[0]).toBeLessThan(p[0].position[0]);expect(q[0].position[1]).toBeLessThan(p[0].position[1]);
 await page.goto('/monitor?camera=1&time=.8');await page.getByRole('button',{name:'Люди',exact:true}).click();
 await expect(page.getByRole('button',{name:`${p[0].global_label} на карте`})).toBeVisible();await expect(page.locator('.floor-zone')).toHaveCount(0);
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/first-exit-map.png',fullPage:true});
 await page.getByRole('button',{name:'Камеры',exact:true}).click();
 await expect.poll(()=>page.locator('.camera-grid video').evaluateAll((vs:HTMLVideoElement[])=>vs.every(v=>v.readyState>=2&&Math.abs(v.currentTime-.8)<.2)),{timeout:30000}).toBeTruthy();
 await expect.poll(()=>page.locator('.camera-grid canvas').evaluateAll((cs:HTMLCanvasElement[])=>cs.slice(0,2).every(c=>{const pixels=c.getContext('2d')!.getImageData(0,0,c.width,c.height).data;let count=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])count++;return count>150})),{timeout:15000}).toBeTruthy();
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/first-exit-grid.png',fullPage:true});
});
