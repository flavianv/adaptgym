/* Saved playback only. Live chat and live replay continue to use the Luna API. */
const savedLabels={luna:'Luna',baseline:'Qwen 4B zero-shot',lora:'Qwen 4B fine-tuned'};
let playTimer=null,rightEvaluation=null,savedBundle=null,liveSchemaVersions={};
const originalReset=reset,originalBusy=busyControls,originalRefresh=refreshReplaySamples,originalLoad=loadReplay;
const rightPanel=document.querySelector('.layout>aside').cloneNode(true);
rightPanel.id='rightPanel';rightPanel.hidden=true;
for(const item of rightPanel.querySelectorAll('[id]'))item.id='right-'+item.id;
document.querySelector('.layout').append(rightPanel);
function playerButton(id,label){
 const paths={Play:'<path d="m8 5 11 7-11 7z" fill="currentColor"/>',Pause:'<path d="M8 5v14M16 5v14"/>','Previous turn':'<path d="M6 5v14M19 5 8 12l11 7z"/>','Next turn':'<path d="M18 5v14M5 5l11 7-11 7z"/>',Reset:'<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>'};
 const button=$(id);button.classList.add('player-icon');button.setAttribute('aria-label',label);button.title=label;
 button.innerHTML='<svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+paths[label]+'</svg>';
}
function stopPlayback(){if(playTimer)clearInterval(playTimer);playTimer=null;if($('mode').value==='saved')playerButton('startExtraction','Play')}
function configureModels(){
 const saved=$('mode').value==='saved',compare=saved&&$('compareModels').checked;
 if(saved){playerButton('startExtraction',playTimer?'Pause':'Play');playerButton('previousTurn','Previous turn');playerButton('nextTurn','Next turn');playerButton('resetPlayback','Reset')}else{for(const id of ['startExtraction','nextTurn']){$(id).classList.remove('player-icon');$(id).removeAttribute('aria-label');$(id).removeAttribute('title')}}
 $('resetPlayback').hidden=!saved;$('turnSeek').hidden=!saved;$('savedModelControls').hidden=!saved;$('previousTurn').hidden=!saved;
 $('rightModelLabel').hidden=!compare;rightPanel.hidden=!compare;
 document.querySelector('.layout').classList.toggle('comparing',compare);
 $('leftModelLabel').firstChild.textContent=compare?'Left model ':'Model ';
 for(const row of document.querySelectorAll('.comparison-turn-metrics,.trace-summary'))row.hidden=!saved;
 for(const label of document.querySelectorAll('#extractorLabel,#right-extractorLabel'))label.classList.toggle('current-turn',saved);
 if(!saved){document.querySelector('.metrics-section h2').textContent='This conversation';$('extractorPrompt').textContent='Simple intent extraction · Luna';$('extractorTitle').textContent='Luna extractor';$('extractorLabel').textContent='02 / LIVE EXTRACTION'}
}
reset=function(){stopPlayback();originalReset();rightEvaluation=null;configureModels()};
busyControls=function(value){
 originalBusy(value);
 if($('mode').value==='saved'){
  $('startExtraction').disabled=value||!replayData||!replayResults.length;
  $('previousTurn').disabled=value||!replayData||replayIndex===0;
  $('turnSeek').disabled=value||!replayResults.length;
  $('resetPlayback').disabled=value||!replayResults.length;
 }
 for(const id of ['leftModel','rightModel','compareModels'])$(id).disabled=value;
};
refreshReplaySamples=async function(){
 if($('mode').value!=='saved')return originalRefresh();
 const response=await fetch('/api/comparison');if(!response.ok)throw Error('Saved comparisons unavailable');
 const index=await response.json();liveSchemaVersions=index.live_schema_versions||{};$('replaySample').replaceChildren();
 for(const entry of index.cases.filter(e=>e.vertical===$('catalog').value)){
  const option=el('option',entry.label);option.value=entry.id;$('replaySample').append(option);
 }
 $('savedResultsMode').disabled=!index.cases.length;
 if($('replaySample').options.length)await loadReplay();
 else $('status').textContent='No saved comparison conversations available.';
};
function readyCheckpoint(evaluation,index){return {...evaluation.checkpoints[index],extractor_prompt:evaluation.extractor_prompt}}
function panelMetrics(cp,scope){
 const target=scope||document.querySelector('.layout>aside');let row=target.querySelector('.comparison-turn-metrics');
 if(!row){row=el('p',undefined,'comparison-turn-metrics');target.querySelector('.panel-body').prepend(row)}
 const format=n=>n===null||n===undefined?'—':(100*n).toFixed(1)+'%';
 const f=cp.feature_score;
 row.className='comparison-turn-metrics '+(f.accepted?'turn-pass':'turn-fail');
 row.replaceChildren(el('strong','Turn '+cp.turn+' · '+(f.accepted?'PASS':'FAIL')+' · '+f.error_count+' error'+(f.error_count===1?'':'s')),el('span','Warnings '+f.warning_count+' · Precision '+format(f.precision)+' · Recall '+format(f.recall),'score-details'));
 if($('compareModels').checked&&rightEvaluation){
  const peer=(scope?replayEvaluation:rightEvaluation).checkpoints[replayIndex].feature_score;
  row.append(el('span',f.error_count<peer.error_count?'Fewer errors this turn':f.error_count>peer.error_count?'More errors this turn':'Same error count this turn','turn-comparison'));
 }
}
function renderTraceSummary(evaluation,scope){
 const target=scope||document.querySelector('.layout>aside'),body=target.querySelector('.panel-body');
 let box=target.querySelector('.trace-summary');if(!box)box=el('section',undefined,'trace-summary');
 const s=evaluation.feature_summary,n=s.turns,bad=evaluation.checkpoints.filter(cp=>!cp.feature_score.accepted),warn=evaluation.checkpoints.filter(cp=>cp.feature_score.warning_count);
 box.replaceChildren();box.hidden=false;box.className='trace-summary'+(bad.length?' has-differences':'');
 box.append(el('h3','Whole trace · '+n+' turns'),el('p',bad.length?bad.length+' / '+n+' turns have feature errors':'All turns pass · every target feature/value is present'));
 const fmt=x=>x===null||x===undefined?'—':(100*x).toFixed(1)+'%';
 box.append(el('p','Value precision '+fmt(s.precision)+' · Recall '+fmt(s.recall)),
  el('p','Passed '+s.passed_turns+'/'+n+' · Errors '+s.errors+' · Warnings '+s.warnings));
 for(const [items,label] of [[bad,'Inspect error turns'],[warn,'Inspect warning turns']])if(items.length){
  const details=el('details');details.append(el('summary',label));
  for(const cp of items){const button=el('button','Turn '+cp.turn);button.type='button';button.onclick=()=>{stopPlayback();replayIndex=cp.turn-1;showSavedTurn()};details.append(button)}
  box.append(details);
 }
 box.append(el('p','Exact feature, operator and value; strings lowercased. Missing/changed targets fail. Extras warn only. Other state diagnostics do not gate this check.','trace-note'));
 const savedVersion=savedBundle?.sample.plan.vertical_schema['x-adaptgym'].version,currentVersion=liveSchemaVersions[evaluation.vertical];
 if(currentVersion&&savedVersion!==currentVersion)box.append(el('p','Historical extraction: '+evaluation.vertical+' schema v'+savedVersion+'. New conversations use v'+currentVersion+'. Scores retain the original target schema.','trace-note'));
 body.prepend(box);
}

function showSavedTurn(){
 if(!replayData||!replayResults.length)return;
 const turn=replayIndex+1;$('conversation').replaceChildren();let current;
 for(const step of replayData.plan.exchanges.slice(0,turn))for(const id of [step.shopper_message_id,step.assistant.message_id]){
  const message=replayData.result.conversation.find(m=>m.id===id);if(!message)continue;
  const line=appendMessage(message.role,message.text);if(id===replayData.plan.exchanges[replayIndex].shopper_message_id)current=line;
 }
 if(current)$('conversation').scrollTop=Math.max(0,current.offsetTop-$('conversation').offsetTop-12);
 renderReplayResult(replayResults[replayIndex]);panelMetrics(replayResults[replayIndex]);renderTraceSummary(replayEvaluation);
 $('extractorTitle').textContent=savedLabels[$('leftModel').value];$('extractorLabel').textContent='Turn '+turn+' of '+replayResults.length;
 if($('compareModels').checked&&rightEvaluation){
  const metricRenderer=renderMetrics;profileScope=rightPanel;
  try{
   renderMetrics=()=>{};const cp=readyCheckpoint(rightEvaluation,replayIndex);renderReplayResult(cp);panelMetrics(cp,rightPanel);renderTraceSummary(rightEvaluation,rightPanel);
   $('extractorTitle').textContent=savedLabels[document.getElementById('rightModel').value];$('extractorLabel').textContent='Turn '+turn+' of '+replayResults.length;
  }finally{profileScope=null;renderMetrics=metricRenderer}
 }
 document.querySelector('.metrics-section h2').textContent='Operational metrics · '+savedLabels[$('leftModel').value];
 $('turnSeek').max=replayResults.length;$('turnSeek').value=turn;
 $('replayProgress').textContent='Turn '+turn+' / '+replayResults.length+' · pre-extracted';
 $('status').textContent='Saved model predictions · no API calls · each model carries its own previous valid prediction.';
 busyControls(false);
}
loadReplay=async function(){
 if($('mode').value!=='saved')return originalLoad();
 reset();busyControls(true);
 try{
  const id=$('replaySample').value;
  if(!savedBundle||savedBundle.id!==id){const response=await fetch('/api/comparison?id='+encodeURIComponent(id));if(!response.ok)throw Error('Saved conversation unavailable');savedBundle=await response.json()}
  replayData=savedBundle.sample;replayEvaluation=savedBundle.models[$('leftModel').value].rollout;rightEvaluation=savedBundle.models[$('rightModel').value].rollout;
  replayResults=replayEvaluation.checkpoints.map((_,i)=>readyCheckpoint(replayEvaluation,i));showSavedTurn();
 }catch(error){$('status').textContent=error.message}
 finally{busyControls(false)}
};
$('startExtraction').onclick=()=>{
 if($('mode').value!=='saved')return extractAllTurns();
 if(playTimer){stopPlayback();return}
 if(replayIndex>=replayResults.length-1)replayIndex=0;
 showSavedTurn();playerButton('startExtraction','Pause');
 playTimer=setInterval(()=>{if(replayIndex>=replayResults.length-1){stopPlayback();return}replayIndex++;showSavedTurn();if(replayIndex>=replayResults.length-1)stopPlayback()},2500);
};
$('resetPlayback').onclick=()=>{stopPlayback();replayIndex=0;showSavedTurn()};
$('turnSeek').oninput=()=>{stopPlayback();replayIndex=Number($('turnSeek').value)-1;showSavedTurn()};
$('previousTurn').onclick=()=>{stopPlayback();if(replayIndex>0){replayIndex--;showSavedTurn()}};
$('nextTurn').onclick=()=>{stopPlayback();if(!busy&&replayResults.length>replayIndex+1){replayIndex++;if($('mode').value==='saved')showSavedTurn();else showReplayTurn()}};
$('replaySample').onchange=()=>loadReplay();
for(const id of ['leftModel','rightModel'])$(id).onchange=()=>{stopPlayback();if(savedBundle&&$('mode').value==='saved'){
 const turn=replayIndex;replayEvaluation=savedBundle.models[$('leftModel').value].rollout;rightEvaluation=savedBundle.models[$('rightModel').value].rollout;
 replayResults=replayEvaluation.checkpoints.map((_,i)=>readyCheckpoint(replayEvaluation,i));replayIndex=turn;showSavedTurn();
}};
$('compareModels').onchange=()=>{configureModels();if($('mode').value==='saved')showSavedTurn()};
$('mode').onchange=async()=>{reset();const replay=$('mode').value!=='chat';$('replayControls').hidden=!replay;$('chat').hidden=replay;
 $('modeNote').textContent=$('mode').value==='saved'?'Precomputed results · Play, pause or compare any two saved models.':replay?'Live extraction uses the Luna API.':'Live conversation uses the Luna API.';
 if(replay)try{await refreshReplaySamples()}catch(error){$('status').textContent=error.message}
};
configureModels();
if(!new URLSearchParams(location.search).has('mode')||new URLSearchParams(location.search).get('mode')==='saved'){$('mode').value='saved';$('mode').onchange()}
