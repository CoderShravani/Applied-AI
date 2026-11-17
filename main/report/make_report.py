"""Build the four-page, black-text engineering report with charts and diagrams."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, KeepTogether, Flowable)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart

ROOT=Path(__file__).parent
OUT=ROOT/'report.pdf'
BLACK=colors.black
MID=colors.HexColor('#555555')
LIGHT=colors.HexColor('#EAEAEA')
WHITE=colors.white

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Cover', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=20, leading=22, alignment=TA_CENTER, textColor=BLACK, spaceAfter=3))
styles.add(ParagraphStyle(name='Deck', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=10, alignment=TA_CENTER, textColor=BLACK, spaceAfter=7))
styles.add(ParagraphStyle(name='H1b', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=12, leading=14, textColor=BLACK, spaceBefore=3, spaceAfter=4))
styles.add(ParagraphStyle(name='H2b', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=9, leading=10.5, textColor=BLACK, spaceBefore=3, spaceAfter=2))
styles.add(ParagraphStyle(name='Bodyb', parent=styles['BodyText'], fontName='Helvetica', fontSize=7.55, leading=9.3, textColor=BLACK, spaceAfter=3))
styles.add(ParagraphStyle(name='Smallb', parent=styles['BodyText'], fontName='Helvetica', fontSize=6.5, leading=7.8, textColor=BLACK, spaceAfter=1.5))
styles.add(ParagraphStyle(name='Metric', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=12, leading=13, alignment=TA_CENTER, textColor=BLACK))
styles.add(ParagraphStyle(name='MetricLabel', parent=styles['BodyText'], fontName='Helvetica', fontSize=6.4, leading=7.2, alignment=TA_CENTER, textColor=BLACK))

def P(s, style='Bodyb'):
    return Paragraph(s, styles[style])

def table(data, widths, header=True, fs=6.5):
    cells=[]
    for ri,row in enumerate(data):
        cells.append([Paragraph(str(x), styles['Smallb']) for x in row])
    t=Table(cells, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    cmds=[('GRID',(0,0),(-1,-1),0.35,BLACK),('VALIGN',(0,0),(-1,-1),'TOP'),
          ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
          ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]
    if header: cmds.append(('BACKGROUND',(0,0),(-1,0),LIGHT))
    t.setStyle(TableStyle(cmds)); return t

class Workflow(Flowable):
    def __init__(self): super().__init__(); self.width=180*mm; self.height=21*mm
    def draw(self):
        labels=['DATA + SPLITS','QLoRA ADAPTER','HELD-OUT EVAL','PAGE-AWARE RAG','LOCAL BENCHMARK','DEPLOYMENT DESIGN']
        n=len(labels); gap=2.2*mm; bw=(self.width-gap*(n-1))/n; y=5*mm; h=11*mm
        for i,label in enumerate(labels):
            x=i*(bw+gap)
            self.canv.setStrokeColor(BLACK); self.canv.setLineWidth(.65); self.canv.rect(x,y,bw,h,stroke=1,fill=0)
            self.canv.setFont('Helvetica-Bold',5.5); self.canv.setFillColor(BLACK)
            words=label.split(' ')
            if len(words)>1:
                self.canv.drawCentredString(x+bw/2,y+6.5*mm,words[0])
                self.canv.drawCentredString(x+bw/2,y+3.8*mm,' '.join(words[1:]))
            else: self.canv.drawCentredString(x+bw/2,y+5*mm,label)
            if i<n-1:
                self.canv.line(x+bw,y+h/2,x+bw+gap,y+h/2)
                self.canv.line(x+bw+gap-1.1*mm,y+h/2+0.8*mm,x+bw+gap,y+h/2)
                self.canv.line(x+bw+gap-1.1*mm,y+h/2-0.8*mm,x+bw+gap,y+h/2)

class BarChart(Flowable):
    def __init__(self, title, cats, values, maxval, suffix='', width=82*mm, height=43*mm):
        super().__init__(); self.width=width; self.height=height; self.title=title; self.cats=cats; self.values=values; self.maxval=maxval; self.suffix=suffix
    def draw(self):
        c=self.canv; c.setFillColor(BLACK); c.setFont('Helvetica-Bold',8); c.drawString(0,self.height-8,self.title)
        left=17*mm; bottom=10*mm; plotw=self.width-left-3*mm; ploth=self.height-22*mm
        c.setStrokeColor(BLACK); c.setLineWidth(.45); c.line(left,bottom,left,bottom+ploth); c.line(left,bottom,left+plotw,bottom)
        for tick in range(0,5):
            y=bottom+ploth*tick/4; val=self.maxval*tick/4
            c.setStrokeColor(colors.HexColor('#BBBBBB')); c.line(left,y,left+plotw,y)
            c.setFillColor(BLACK); c.setFont('Helvetica',5.8); c.drawRightString(left-2,y-2,f'{val:.0f}{self.suffix}')
        bw=plotw/(len(self.values)*1.8)
        for i,(cat,val) in enumerate(zip(self.cats,self.values)):
            x=left+(i+.5)*plotw/len(self.values)-bw/2; h=ploth*val/self.maxval
            c.setFillColor(colors.HexColor('#777777') if i else BLACK); c.rect(x,bottom,bw,h,stroke=1,fill=1)
            c.setFillColor(BLACK); c.setFont('Helvetica-Bold',6.4); c.drawCentredString(x+bw/2,bottom+h+2,f'{val:g}{self.suffix}')
            c.setFont('Helvetica',6); c.drawCentredString(x+bw/2,bottom-7,cat)

class Donut(Flowable):
    def __init__(self): super().__init__(); self.width=82*mm; self.height=43*mm
    def draw(self):
        c=self.canv; c.setFillColor(BLACK); c.setFont('Helvetica-Bold',8); c.drawString(0,self.height-8,'Fine-tuned exact-match outcomes (100 tests)')
        x=24*mm; y=21*mm; r=12*mm
        c.setFillColor(BLACK); c.wedge(x-r,y-r,x+r,y+r,0,270,stroke=1,fill=1)
        c.setFillColor(colors.HexColor('#BDBDBD')); c.wedge(x-r,y-r,x+r,y+r,270,90,stroke=1,fill=1)
        c.setFillColor(WHITE); c.circle(x,y,6*mm,stroke=0,fill=1)
        c.setFillColor(BLACK); c.setFont('Helvetica-Bold',8); c.drawCentredString(x,y-2,'75%')
        c.setFont('Helvetica',7); c.drawString(48*mm,25*mm,'75 exact matches')
        c.drawString(48*mm,17*mm,'25 remaining cases')
        c.setStrokeColor(BLACK); c.line(44*mm,27*mm,46*mm,27*mm); c.line(44*mm,19*mm,46*mm,19*mm)

story=[]
# PAGE 1
story += [P('SMALL MODELS, REAL SYSTEMS','Cover'),P('A measured engineering study of fine-tuning, retrieval, and local inference','Deck'),
          P('<b>Shravani Kulkarni</b>  |  17 November 2026  |  Three tasks • one reproducible repository','Deck'),
          Workflow(),Spacer(1,2),P('Executive summary','H1b'),
          P('Built and evaluated three compact-model components: a QLoRA adapter for strict request-to-JSON extraction, a page-aware RAG pipeline over the NIST AI Risk Management Framework, and an Ollama quantization benchmark. The work produced an adapter, evaluated 100 held-out examples, tested 15 retrieval questions at two chunk sizes, and benchmarked 20 local model responses.'),
          P('1  |  QLoRA fine-tuning: natural language → JSON','H1b'),
          P('<b>Base:</b> Qwen/Qwen2.5-0.5B-Instruct  •  <b>Method:</b> 4-bit NF4 QLoRA  •  <b>GPU:</b> Tesla T4<br/><b>Data:</b> 800 examples (600 train / 100 validation / 100 held-out test). Five-field output contract: action, object, priority, due_in_days, assignee.'),
          P('<b>Configuration:</b> rank 8, alpha 16, dropout 0.05; q_proj/k_proj/v_proj/o_proj; learning rate 2e-4; 3 epochs; seed 42; micro-batch 4 × accumulation 4 (effective batch 16); sequence length 256; bfloat16 compute.'),
          table([['Held-out metric (n=100)','Base','QLoRA'],['JSON schema validity','0%','<b>100%</b>'],['Exact five-field match','0%','<b>75%</b>'],['Action / object accuracy','Not scoreable*','99% / 96%'],['Priority / deadline accuracy','Not scoreable*','83% / 96%'],['Assignee accuracy','Not scoreable*','<b>100%</b>']],[65*mm,36*mm,45*mm]),
          Spacer(1,2),
          P('*The evaluator scores fields only when the complete prediction passes schema validation. The base produced no schema-valid outputs; its field-level scores are therefore not interpretable. The adapted model returned schema-valid JSON for all 100 test inputs and exactly matched all five fields in 75 cases.' ,'Smallb'),
          Donut(),
          P('<b>Observed fine-tuned edge cases:</b> “tomorrow” was returned as day 0 instead of day 1; “work item” mapped to ticket instead of task; and “there is no due date” mapped to day 0 instead of -1. Targeted refinements: relative-date normalization, object-synonym hard negatives, and explicit no-deadline examples. Full examples and fixes are in the result JSON.','Smallb'),PageBreak()]
# PAGE 2
story += [P('2  |  Retrieval from scratch: NIST AI RMF','H1b'),
          P('<b>Corpus:</b> NIST AI Risk Management Framework 1.0, 48 PDF pages (January 2023). <b>Evaluation:</b> 15 authored questions, top-5 retrieval, 60-word overlap, 350- and 700-word chunks. Embeddings use sentence-transformers/all-MiniLM-L6-v2; normalized vectors are ranked with NumPy cosine similarity and combined with custom BM25 (0.75 dense / 0.25 lexical). No LangChain or LlamaIndex.'),
          table([['Chunk size','Questions','Recall@5','Gold-term coverage@5'],['350 words','15','60.0% (9/15)','100% (15/15)'],['700 words','15','<b>66.7% (10/15)</b>','100% (15/15)']],[36*mm,28*mm,45*mm,48*mm]),
          Spacer(1,3),
          BarChart('Gold-page retrieval rate', ['350 words','700 words'], [60,66.7],100,'%',width=85*mm,height=48*mm),
          P('<b>Interpretation.</b> The 700-word configuration returned one additional gold-page hit, a 6.7 percentage-point gain, while both chunk sizes covered at least one expected gold term for all 15 questions. For this evaluation set, 700 words is the stronger starting point; 350-word chunks remain useful where tighter passage boundaries are preferred.'),
          P('Retrieval diagnostics and targeted improvements','H2b'),
          table([['Question','Observed top-5 page IDs (350 / 700)','Engineering response'],
                 ['Q10 — intended purpose of AI RMF','[7, 11, 40, 30, 6] / [7, 40, 6, 43, 42]','Expand query with “intended use”, “trustworthiness”, and design/development/use/evaluation terminology; rerank candidates.'],
                 ['Q13 — action when risk cannot be mitigated','[10, 12, 12, 47, 21] / [10, 12, 47, 27, 21]','Add residual-risk, response, unacceptable-impact, and cease-development/deployment terms; rerank top candidates.']],[33*mm,58*mm,66*mm]),
          Spacer(1,2),
          P('<b>Why hybrid retrieval:</b> embeddings recover semantic matches across paraphrases; BM25 rewards exact acronyms, section names, and rare terms. A cross-encoder reranker compares query and candidate passage jointly, improving ordering at additional compute cost.'),
          P('<b>Pipeline:</b> PDF extraction → page-aware overlapping chunks → normalized embeddings → dense + BM25 ranking → top-k evidence → grounded response with page/chunk references.','Smallb'),PageBreak()]
# PAGE 3
story += [P('3  |  Local inference: quantization and quality','H1b'),
          P('<b>Runtime:</b> Ollama / llama.cpp runner  •  <b>GPU:</b> NVIDIA Tesla T4 (15,360 MiB)  •  <b>RAM:</b> 12 GiB<br/>Ten identical prompts per model, warm-up enabled, temperature 0.2, 256-token output cap, 4,096 context reported by Ollama; 20 measured responses total.'),
          table([['Setting','Verified Ollama tag','Format','Mean tokens/s','Mean wall time'],['Q4','qwen2.5:1.5b','Q4_K_M','<b>136.81</b>','1.691 s'],['Q8','qwen2.5:1.5b-instruct-q8_0','Q8_0','99.31','2.078 s']],[18*mm,65*mm,23*mm,31*mm,30*mm]),
          Spacer(1,3),
          BarChart('Mean generation throughput', ['Q4_K_M','Q8_0'], [136.81,99.31],160,' tok/s',width=85*mm,height=48*mm),
          P('<b>Performance:</b> Q4 delivered 37.8% higher mean generation throughput in this run. The Q4 and Q8 model tags and quantization labels were verified with `ollama show`.'),
          P('Blinded rubric review (10 prompts × 2 models)','H2b'),
          table([['Criterion (1–5)','Q4_K_M','Q8_0'],['Correctness','3.40','3.40'],['Completeness','3.10','3.20'],['Instruction-following','3.60','3.70'],['Clarity','3.80','4.10'],['Overall mean','3.48','<b>3.60</b>']],[70*mm,38*mm,38*mm]),
          P('The rubric review shows a close trade-off: Q4 is faster; Q8 scores slightly higher on completeness, instruction-following, and clarity, with correctness tied. These scores describe this ten-prompt comparison rather than a universal ranking.'),
          P('Observed memory snapshot','H2b'),
          table([['Observation','Measured value'],['Both models resident','Ollama reported 100% GPU placement for each'],['Ollama resident sizes','Q4 1.2 GB  •  Q8 1.8 GB'],['GPU memory, both loaded','3,113 / 15,360 MiB combined snapshot'],['GPU process entries','1,240 MiB and 1,870 MiB'],['System RAM','2.5 GiB used; 10 GiB available']],[63*mm,83*mm]),
          P('Memory was sampled with both models loaded simultaneously. The 3,113 MiB figure is a combined snapshot, not isolated per-model peak usage.','Smallb'),PageBreak()]
# PAGE 4
story += [P('4  |  Serving design, memory economics, and conclusions','H1b'),
          P('20 connected users on one 24 GB GPU','H2b'),
          P('<b>Assumptions:</b> 24 GB GPU; reserve 2 GB for CUDA/runtime/fragmentation; 1.5 GB illustrative 1.5B 4-bit model-plus-runtime budget; 1,000 prompt tokens and 200 generated tokens per session; FP16 KV cache; 20 active sessions, not necessarily 20 simultaneous decode streams.'),
          table([['Illustrative GPU budget','GB'],['GPU capacity','24.0'],['Runtime / fragmentation reserve','−2.0'],['Quantized model + runtime','−1.5'],['KV cache + activations + workspace budget','<b>20.5</b>']],[100*mm,45*mm]),
          P('<b>Serving plan:</b> one continuous-batching server where supported (otherwise Ollama/llama.cpp with bounded parallelism); begin at eight active sequences; cap context and output; use prefix caching; place an API gateway in front with queueing, per-user limits, timeouts, cancellation, and back-pressure. Monitor p50/p95 time-to-first-token, inter-token latency, throughput, queue depth, GPU memory, and error rate. Queue excess arrivals rather than promising 20-way low-latency decoding.'),
          P('<b>Design status:</b> 20 connected users with controlled concurrency and queueing is the target architecture. A 20-user load test has not been performed; capacity is not claimed as verified.' ,'Smallb'),
          P('7B parameter-state estimate','H2b'),
          table([['Method','Approx. state','Breakdown'],['Full mixed-precision AdamW','<b>112 GB</b>','14 GB FP16 weights + 28 GB FP32 master weights + 14 GB FP16 gradients + 56 GB Adam moments'],['LoRA','<b>14.1 GB</b>','14 GB frozen FP16 base + ~0.11 GB adapter state (7M parameters × ~16 bytes)'],['QLoRA','<b>4.31 GB</b>','3.5 GB 4-bit base + 0.7 GB metadata allowance + ~0.11 GB adapter state']],[38*mm,30*mm,77*mm]),
          P('These are parameter-state estimates in decimal GB, before activations, temporary buffers, attention workspaces, context-dependent KV cache, and fragmentation. Practical peak memory is higher.' ,'Smallb'),
          P('Engineering conclusion','H2b'),
          P('The three experiments demonstrate complementary strengths: QLoRA can make a compact model reliably follow a constrained output contract; hybrid retrieval can surface evidence from a long standards document; and quantization creates a measurable speed/quality trade-off on local hardware. The repository includes task code, configurations, measured result summaries, failure diagnostics, and this four-page report.'),
          P('<b>Reproduction anchors:</b> 800 training examples; 100 held-out evaluations; 15 RAG questions; 20 inference responses; 48-page source; Tesla T4 GPU; verified Q4_K_M and Q8_0 model tags.','Smallb'),
          Spacer(1,4),P('<b>Source:</b> NIST AI Risk Management Framework 1.0, NIST AI 100-1 (January 2023), https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf','Smallb')]

def footer(canvas, doc):
    canvas.saveState(); canvas.setFillColor(BLACK); canvas.setFont('Helvetica',6)
    canvas.drawString(13*mm,7*mm,'SMALL MODELS, REAL SYSTEMS  |  ENGINEERING REPORT')
    canvas.drawRightString(A4[0]-13*mm,7*mm,f'{doc.page} / 4')
    canvas.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=13*mm,leftMargin=13*mm,topMargin=10*mm,bottomMargin=12*mm,title='Small Models, Real Systems',author='Shravani Kulkarni')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(f'Created {OUT}')
