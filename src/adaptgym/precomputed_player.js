/* Saved playback only. Live chat and live replay continue to use the Luna API. */
const savedLabels={luna:'Luna',baseline:'Qwen 4B zero-shot',lora:'Qwen 4B fine-tuned'};
let playTimer=null,rightEvaluation=null,savedBundle=null;
const originalReset=reset,originalBusy=busyControls,originalRefresh=refreshReplaySamples,originalLoad=loadReplay;
const rightPanel=document.querySelector('.layout>aside').cloneNode(true);
rightPanel.id='rightPanel';rightPanel.hidden=true;
for(const item of rightPanel.querySelectorAll('[id]'))item.id='right-'+item.id;
document.querySelector('.layout').append(rightPanel);
function stopPlayback(){if(playTimer)clearInterval(playTimer);playTimer=null;if($('mode').value==='saved')$('startExtraction').textContent='Play'}
function configureModels(){
 const saved=$('mode').value==='saved',compare=saved&&$('compareModels').checked;
 if(saved)$('startExtraction').textContent=playTimer?'Pause':'Play';
 $('savedModelControls').hidden=!saved;$('previousTurn').hidden=!saved;
 $('rightModelLabel').hidden=!compare;rightPanel.hidden=!compare;
 document.querySelector('.layout').classList.toggle('comparing',compare);
 $('leftModelLabel').firstChild.textContent=compare?'Left model ':'Model ';
 for(const row of document.querySelectorAll('.comparison-turn-metrics'))row.hidden=!saved;
 if(!saved){$('extractorPrompt').textContent='Simple intent extraction · Luna';$('extractorTitle').textContent='Luna extractor';$('extractorLabel').textContent='02 / LIVE EXTRACTION'}
}
reset=function(){stopPlayback();originalReset();rightEvaluation=null;configureModels()};
busyControls=function(value){
 originalBusy(value);
 if($('mode').value==='saved'){
  $('startExtraction').disabled=value||!replayData||!replayResults.length;
  $('previousTurn').disabled=value||!replayData||replayIndex===0;
 }
 for(const id of ['leftModel','rightModel','compareModels'])$(id).disabled=value;
};
refreshReplaySamples=async function(){
 if($('mode').value!=='saved')return originalRefresh();
 const response=await fetch('/api/comparison');if(!response.ok)throw Error('Saved comparisons unavailable');
 const index=await response.json();$('replaySample').replaceChildren();
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
 row.textContent='Precision '+format(cp.metrics?.value_precision)+' · Recall '+format(cp.metrics?.value_recall)+' · '+(cp.score.valid?'Valid state':'Invalid state');
}
function showSavedTurn(){
 if(!replayData||!replayResults.length)return;
 const turn=replayIndex+1;$('conversation').replaceChildren();let current;
 for(const step of replayData.plan.exchanges.slice(0,turn))for(const id of [step.shopper_message_id,step.assistant.message_id]){
  const message=replayData.result.conversation.find(m=>m.id===id);if(!message)continue;
  const line=appendMessage(message.role,message.text);if(id===replayData.plan.exchanges[replayIndex].shopper_message_id)current=line;
 }
 if(current)$('conversation').scrollTop=Math.max(0,current.offsetTop-$('conversation').offsetTop-12);
 renderReplayResult(replayResults[replayIndex]);panelMetrics(replayResults[replayIndex]);
 $('extractorTitle').textContent=savedLabels[$('leftModel').value];$('extractorLabel').textContent='SAVED EXTRACTION';
 if($('compareModels').checked&&rightEvaluation){
  const metricRenderer=renderMetrics;profileScope=rightPanel;
  try{
   renderMetrics=()=>{};const cp=readyCheckpoint(rightEvaluation,replayIndex);renderReplayResult(cp);panelMetrics(cp,rightPanel);
   $('extractorTitle').textContent=savedLabels[document.getElementById('rightModel').value];$('extractorLabel').textContent='SAVED EXTRACTION';
  }finally{profileScope=null;renderMetrics=metricRenderer}
 }
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
 showSavedTurn();$('startExtraction').textContent='Pause';
 playTimer=setInterval(()=>{if(replayIndex>=replayResults.length-1){stopPlayback();return}replayIndex++;showSavedTurn();if(replayIndex>=replayResults.length-1)stopPlayback()},2500);
};
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
if(new URLSearchParams(location.search).get('mode')==='saved'){$('mode').value='saved';$('mode').onchange()}
