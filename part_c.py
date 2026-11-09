"""Part C: privacy-preserving support ticket classification using supplied offline mock."""
from pathlib import Path
import re, json, random, time
import pandas as pd
import mock_llm

OUT=Path('outputs'); OUT.mkdir(exist_ok=True)
MAX_INPUT_TOKENS=400
BASE_DELAY=.05
ALLOWED={'billing','delivery','technical','fraud'}
PATTERNS=[('email',re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'),'[EMAIL]'),('phone',re.compile(r'\b[6-9]\d{9}\b'),'[PHONE]'),('order_id',re.compile(r'ORD-\d{6}'),'[ORDER_ID]')]
ZERO='CLASSIFY the support ticket into exactly one of billing, delivery, technical, fraud. Return JSON only: {"category": "...", "reason": "..."}. Never invent a category.'
FEW=ZERO+'\nEXAMPLES:\n"Unauthorized card transaction" => {"category":"fraud","reason":"unauthorized transaction"}\n"My rider is late" => {"category":"delivery","reason":"late delivery"}\n"App freezes on login" => {"category":"technical","reason":"app failure"}'

# Remove private information *before* text goes into any model message.
def redact(text):
    counts={}
    for kind,pattern,replacement in PATTERNS:
        text,counts[kind]=pattern.subn(replacement,text)
    assert all(not p.search(text) for _,p,_ in PATTERNS)
    return text,counts

# Try again after temporary model failures, with a short increasing delay.
def retry_chat(messages,ticket_id):
    for retry in range(4): # initial attempt + 3 retries
        try:return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry==3: raise
            wait=BASE_DELAY*(2**retry)*random.uniform(.9,1.15)
            print(f'ticket={ticket_id} retry={retry} wait={wait:.5f}s')
            time.sleep(wait)

# Accept valid JSON or a single clearly mentioned category; otherwise request review.
def parse_answer(content):
    try:
        obj=json.loads(content)
    except (TypeError,ValueError):
        terms=set(re.findall(r'[a-z]+',(content or '').lower())) & ALLOWED
        return ('classified_keyword',next(iter(terms))) if len(terms)==1 else ('flagged_invalid_output','')
    category=obj.get('category') if isinstance(obj,dict) else None
    return ('classified_json',category) if category in ALLOWED else ('flagged_invalid_output','')

# Run all tickets in their original order and track the token budget.
def run(tickets,segments,prompt,budget=None):
    mock_llm.reset()
    result=[]; total_tokens=0; counts=dict.fromkeys(['email','phone','order_id'],0); retries_before=mock_llm._state['calls']
    for row in tickets.itertuples(index=False):
        raw=row.text
        redacted,ct=redact(raw)
        for kind,n in ct.items():counts[kind]+=n
        tokens=0; category=''
        if not raw.strip(): status='skipped_empty'
        elif mock_llm.est_tokens(raw)>MAX_INPUT_TOKENS: status='flagged_long'
        elif budget is not None and total_tokens>=budget: status='budget_exhausted'
        else:
            # Only the redacted text is placed in the user message.
            messages=[{'role':'system','content':prompt},{'role':'user','content':redacted}]
            assert messages[1]['content']==redacted
            assert all(not p.search(messages[1]['content']) for _,p,_ in PATTERNS)
            response=retry_chat(messages,row.ticket_id)
            tokens=response.usage.total_tokens
            total_tokens+=tokens
            status,category=parse_answer(response.choices[0].message.content)
        info=segments.get(row.customer_id,{})
        is_premium=bool(info.get('is_premium',False))
        segment_name=info.get('segment_name','')
        if status not in ('classified_json','classified_keyword'):
            priority,route='NORMAL','human_review'
            category=''
        elif category=='fraud': priority,route='HIGH','security_team'
        elif is_premium: priority,route='HIGH','premium_desk'
        else: priority,route='NORMAL','support_queue'
        result.append({'ticket_id':row.ticket_id,'customer_id':row.customer_id,'status':status,'category':category,'segment_name':segment_name,'priority':priority,'route':route,'tokens_used':tokens,'true_category':row.true_category})
    frame=pd.DataFrame(result)
    classified=frame.status.isin(['classified_json','classified_keyword'])
    actual=(frame.loc[classified,'category']==frame.loc[classified,'true_category']).sum()
    report={'statuses':frame.status.value_counts().to_dict(),'classified':int(classified.sum()),'correct':int(actual),'accuracy':round(float(actual/max(1,classified.sum())),4),'total_tokens':int(total_tokens),'tokens_per_classified':round(total_tokens/max(1,classified.sum()),2),'redactions':counts,'redactions_total':sum(counts.values()),'retries':int(mock_llm._state['calls']-mock_llm._state['ok'])}
    return frame,report

def main():
    tickets=pd.read_csv('data/raw/tickets.csv',keep_default_na=False)
    segments=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')[['segment_name','is_premium']].to_dict('index')
    zero,z_report=run(tickets,segments,ZERO)
    pd.crosstab(zero.true_category,zero.category).to_csv(OUT/'zero_shot_confusion.csv')
    few,f_report=run(tickets,segments,FEW)
    few[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(OUT/'triage_results.csv',index=False)
    demo,b_report=run(tickets,segments,ZERO,budget=6000)
    demo[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(OUT/'triage_budget_demo.csv',index=False)
    report={'zero_shot':z_report,'few_shot':f_report,'budget_demo':b_report,'budget_used':6000}
    (OUT/'part_c_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert len(few)==164 and f_report['classified']==147
    assert f_report['statuses']['classified_json']==135 and f_report['statuses']['classified_keyword']==12 and f_report['statuses']['flagged_invalid_output']==9
    assert b_report['statuses'].get('budget_exhausted',0)>0

if __name__=='__main__':main()
