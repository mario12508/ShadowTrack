import {test,expect} from '@playwright/test';

test('real FULL -> camera switching -> playback -> bbox selection -> track detail',async({page,request})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const stats=await (await request.get('/api/stats')).json();expect(stats.frame).toBe(144110);expect(stats.detection).toBe(136020);
 await page.goto('/monitor?camera=3&time=129');
 const video=page.locator('.camera-panel video');
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:20000}).toBeGreaterThanOrEqual(2);
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(128);
 await page.getByRole('button',{name:'Камера 2',exact:true}).click();
 await expect(page.locator('.current-camera-label')).toContainText('02');
 await page.getByRole('button',{name:'Камера 3',exact:true}).click();
 await page.getByRole('slider',{name:'Временная шкала'}).fill('129');
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:20000}).toBeGreaterThan(130);
 await page.getByRole('button',{name:'Пауза',exact:true}).click();
 const t=await video.evaluate((v:HTMLVideoElement)=>v.currentTime);
 const overlay=await (await request.get(`/api/overlays/3?start=${Math.max(0,t-.15)}&end=${t+.1}`)).json();
 const frame=[...overlay.frames].reverse().find((f:any)=>f.media_time<=t+.008&&f.tracks.length);
 expect(frame).toBeTruthy();const target=frame.tracks[0];
 const canvas=page.locator('.camera-panel canvas');const box=(await canvas.boundingBox())!;
 const [x1,y1,x2,y2]=target.bbox_xyxy;
 await canvas.click({position:{x:(x1+x2)/2/2592*box.width,y:(y1+y2)/2/1944*box.height}});
 await expect(page.locator('.track-selection')).toContainText(target.display_id);
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/monitor-desktop.png',fullPage:true});
 await page.getByRole('link',{name:'История',exact:true}).click();
 await expect(page.locator('h2')).toHaveText(target.global_label||target.display_id);
 await expect(page.getByText('Глобальный эпизод',{exact:true})).toBeVisible();
 expect(errors).toEqual([]);
});

test('all source videos use one native organizer clock',async({page})=>{
 await page.goto('/monitor?camera=3&time=129');

 await page.getByRole('button',{name:'Камера 2',exact:true}).click();
 await expect(page.locator('.video-unavailable')).toHaveCount(0);
 await expect(page.locator('.clock-status,.estimated-button,.clock-button')).toHaveCount(0);
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 const video=page.locator('.camera-panel video');
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:20000}).toBeGreaterThan(130);
});

test('designer landing and responsive map have no horizontal overflow',async({page})=>{
 await page.goto('/');await expect(page.locator('h1')).toContainText('Один трек.');
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/landing-desktop.png',fullPage:true});
 await page.setViewportSize({width:393,height:852});
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/landing-mobile.png',fullPage:true});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBeTruthy();
 await page.goto('/monitor?camera=3&time=129');
 await expect(page.locator('.camera-panel')).toBeVisible();
 await expect(page.locator('.current-camera-label')).toContainText('03');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBeTruthy();
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/monitor-mobile.png',fullPage:true});
});

test('all workspace pages use the actual API',async({page})=>{
 for(const [route,heading] of [['/dashboard','Вся история. В одном месте.'],['/tracks','Архив наблюдений'],['/analytics','Движение в цифрах'],['/settings','Время и настройки'],['/exports','Результаты — с собой']]){
  await page.goto(route);await expect(page.locator('h2')).toHaveText(heading);await expect(page.locator('.notice').filter({hasText:'API недоступен'})).toHaveCount(0);
 }
 await expect(page.locator('.download-list a').first()).toBeVisible();
});

test('saved annotated local-ID MP4 decodes through HTTP Range in the React exports page',async({page})=>{
 await page.goto('/exports');
 const video=page.locator('.rendered-preview video');
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:20000}).toBeGreaterThanOrEqual(2);
 await video.evaluate(async(v:HTMLVideoElement)=>{v.currentTime=129;await v.play()});
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime),{timeout:20000}).toBeGreaterThan(130);
 await video.evaluate((v:HTMLVideoElement)=>v.pause());
 await video.screenshot({path:'../private_outputs/mvp_browser_smoke/annotated-mp4-preview.png'});
});

test('all five real camera players advance together under the shared relative playback control',async({page})=>{
 await page.goto('/monitor?camera=3&time=129');
 await page.getByRole('button',{name:'Камеры',exact:true}).click();
 const videos=page.locator('.camera-grid video');await expect(videos).toHaveCount(5);
 await expect.poll(()=>videos.evaluateAll((items:HTMLVideoElement[])=>items.every(v=>v.readyState>=2&&v.currentTime>128)),{timeout:30000}).toBeTruthy();
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 await expect.poll(()=>videos.evaluateAll((items:HTMLVideoElement[])=>items.every(v=>v.currentTime>129.6)),{timeout:20000}).toBeTruthy();
 await page.getByRole('button',{name:'Пауза',exact:true}).click();
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/five-camera-grid.png',fullPage:true});
 await expect(page.locator('.clock-explainer')).toContainText('исходная шкала записей');
});
