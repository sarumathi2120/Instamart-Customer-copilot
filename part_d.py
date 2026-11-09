"""Part D: offline policy retrieval and a checkpointed ReAct-style refund agent."""
from pathlib import Path
from datetime import datetime, timezone
import csv, json, sqlite3, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat

OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
STATE=Path('state'); STATE.mkdir(exist_ok=True)
POLICIES=Path('data/policies')
SYSTEM_POLICY='POLICY ASSISTANT: Answer only from the supplied CONTEXT. If unsupported say I could not find that information in the policy documents.'
SYSTEM_AGENT='AGENT: Use the supplied tools to verify order status, read policy, calculate refund, and issue or escalate only when eligible. Never make up an order. Keep decisions evidence-based.'

# Split policies into 40-word chunks. Overlap prevents facts at boundaries being lost.
def build_index(overlap=10):
    chunks=[]; step=40-overlap
    for filename in sorted(POLICIES.glob('*.txt')):
        words=filename.read_text(encoding='utf-8').split()
        for start in range(0,len(words),step):
            chunks.append({'source':filename.name,'text':' '.join(words[start:start+40])})
            if start+40>=len(words): break
    vectorizer=TfidfVectorizer()
    matrix=vectorizer.fit_transform([r['text'] for r in chunks])
    return chunks,vectorizer,matrix

INDEX=build_index(10)

# Match a question to the most relevant policy chunks using TF-IDF similarity.
def retrieve(query,k=3,index=None):
    chunks,vec,matrix=index if index is not None else INDEX
    query_vec=vec.transform([query])
    scores=cosine_similarity(query_vec,matrix)[0]
    ranked=scores.argsort()[::-1][:k]
    return [{'source':chunks[i]['source'],'score':round(float(scores[i]),6),'text':chunks[i]['text']} for i in ranked]

# Answer only from retrieved policies rather than inventing policy details.
def answer(question,index=None):
    docs=retrieve(question,3,index)
    context='\n'.join(f"[{d['source']}]\n{d['text']}" for d in docs)
    prompt=f'CONTEXT:\n{context}\nQUESTION:\n{question}'
    response=chat([{'role':'system','content':SYSTEM_POLICY},{'role':'user','content':prompt}])
    return response.choices[0].message.content, [d['source'] for d in docs]

def evaluate(index):
    expected=pd.read_csv('data/raw/policy_eval.csv',keep_default_na=False)
    hit=correct=fallback=0; failures=[]
    details=[]
    for row in expected.itertuples(index=False):
        response,sources=answer(row.question,index)
        if row.expected_doc=='NONE':
            fallback+=response.startswith('I could not find')
        else:
            hit+=row.expected_doc in sources
            correct+=row.expected_fact in response
            if row.expected_fact not in response or row.expected_doc not in sources:
                failures.append(row.question)
        details.append({'question':row.question,'expected_doc':row.expected_doc,'expected_fact':row.expected_fact,'sources':'|'.join(sources),'answer':response})
    return {'source_hits':int(hit),'answer_hits':int(correct),'fallback_hits':int(fallback),'failures':failures},pd.DataFrame(details)

# Use a parameterised SQL query; do not concatenate user-provided IDs into SQL.
def get_order(order_id):
    with sqlite3.connect('data/instamart.db') as con:
        row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
    return dict(zip(['order_id','status','order_amount','delivery_minutes'],row)) if row else {'error':'order not found'}

def search_policy(query):return {'results':retrieve(query)}

def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount)
    return {'refund_inr':round(amount,2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}

def _record_action(tool,order_id,refund_inr):
    record={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return {'refund_inr':record['refund_inr']}

def issue_refund(order_id,refund_inr):return _record_action('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return _record_action('escalate_to_human',order_id,refund_inr)

TOOLS_SPEC=[
 {'name':'get_order','description':'Look up an order by identifier in SQLite.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve relevant written policies.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate a late-delivery refund amount.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record a simulated authorised refund in the ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund requiring human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
]
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}

def checkpoint(run_id,scenario_id,messages,tokens,steps,llm_calls):
    data={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':len(steps),'steps':steps,'tools_executed':[s['tool'] for s in steps],'llm_calls':llm_calls}
    (STATE/f'{run_id}.json').write_text(json.dumps(data,indent=2))

def _run(messages,scenario_id,run_id,max_iterations,token_limit,time_limit_seconds,steps=None,tokens_used=0,llm_calls=0,crash_after_tools=None):
    steps=[] if steps is None else list(steps)
    started=time.monotonic()
    def result(status,text):return {'scenario_id':scenario_id,'status':status,'final_answer':text,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used}
    while llm_calls<max_iterations:
        if time.monotonic()-started>=time_limit_seconds:
            return result('FALLBACK_TIMEOUT','Human handoff: agent exceeded its time limit.')
        response=chat(messages,tools=TOOLS_SPEC)
        llm_calls+=1
        tokens_used+=response.usage.total_tokens
        if tokens_used>token_limit:
            return result('FALLBACK_TOKEN_LIMIT','Human handoff: model token budget exhausted.')
        msg=response.choices[0].message
        if not msg.tool_calls:
            final=msg.content or 'Human handoff: no final answer.'
            return result(final.split(':',1)[0],final)
        tc=[{'id':c.id,'name':c.name,'arguments':c.arguments} for c in msg.tool_calls]
        messages.append({'role':'assistant','content':None,'tool_calls':tc})
        for call in tc:
            name=call['name'];args=call['arguments']
            if name not in TOOL_MAP:
                return result('FALLBACK_UNKNOWN_TOOL',f'Human handoff: unknown tool {name}.')
            observation=TOOL_MAP[name](**args)
            steps.append({'tool':name,'arguments':args,'observation':observation})
            messages.append({'role':'tool','tool_call_id':call['id'],'name':name,'content':json.dumps(observation)})
            checkpoint(run_id,scenario_id,messages,tokens_used,steps,llm_calls)
            if crash_after_tools is not None and len(steps)==crash_after_tools:
                raise RuntimeError('Simulated crash immediately after checkpoint')
    return result('FALLBACK_MAX_ITERATIONS','Human handoff: maximum six model iterations reached without a final response.')

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    messages=[{'role':'system','content':SYSTEM_AGENT},{'role':'user','content':user_message}]
    return _run(messages,scenario_id,run_id or scenario_id,max_iterations,token_limit,time_limit_seconds,crash_after_tools=crash_after_tools)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    saved=json.loads((STATE/f'{run_id}.json').read_text())
    return _run(saved['messages'],saved['scenario_id'],run_id,max_iterations,token_limit,time_limit_seconds,saved['steps'],saved['tokens_used'],saved['llm_calls'])

def main():
    chunks,_,_=INDEX
    counts=pd.Series([d['source'] for d in chunks]).value_counts().to_dict()
    standard,sd=evaluate(INDEX)
    none,nd=evaluate(build_index(0))
    sd.to_csv(OUT/'rag_eval_overlap10.csv',index=False)
    nd.to_csv(OUT/'rag_eval_overlap0.csv',index=False)
    scenarios=pd.read_csv('data/raw/agent_scenarios.csv')
    ledger=OUT/'refund_ledger.jsonl';ledger.write_text('')
    runs=OUT/'agent_runs.jsonl';runs.write_text('')
    outcomes=[]
    for row in scenarios.itertuples(index=False):
        actual=run_agent(row.message,row.scenario_id)
        with runs.open('a') as f:f.write(json.dumps(actual)+'\n')
        outcomes.append(actual)
    demo_token=run_agent(scenarios.iloc[0].message,'token_demo',token_limit=200)
    demo_time=run_agent(scenarios.iloc[0].message,'time_demo',time_limit_seconds=0)
    try:
        run_agent(scenarios.iloc[0].message,'S1',run_id='s1_crash_resume',crash_after_tools=2)
    except RuntimeError as e:
        assert 'Simulated crash' in str(e)
    resumed=resume_agent('s1_crash_resume')
    report={'chunk_counts':counts,'overlap_10':standard,'overlap_0':none,
      'scenarios':[{'scenario_id':r['scenario_id'],'status':r['status'],'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'tools':[s['tool'] for s in r['steps']],'final_answer':r['final_answer']} for r in outcomes],
      'token_demo_status':demo_token['status'],'timeout_demo_status':demo_time['status'],
      'resume_status':resumed['status'],'resume_tools':[x['tool'] for x in resumed['steps']]}
    (OUT/'part_d_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert len(chunks)==15,counts
    assert standard['source_hits']==12 and standard['answer_hits']==12 and standard['fallback_hits']==2,standard
    assert [r['status'] for r in outcomes]==['REFUND','REFUND','ESCALATE','DENIED','DENIED','NOT_FOUND','FALLBACK_MAX_ITERATIONS']
    assert demo_token['status']=='FALLBACK_TOKEN_LIMIT' and demo_time['status']=='FALLBACK_TIMEOUT'
    assert resumed['status']=='REFUND' and report['resume_tools']==['get_order','search_policy','calculate_refund','issue_refund']

if __name__=='__main__':main()
