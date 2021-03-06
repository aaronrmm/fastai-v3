from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import re
from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://www.dropbox.com/s/dctv4w1ekubmtsk/export.pkl?dl=1'
export_file_name = 'export.pkl'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    text = "GM : "+data['file']+data['file']+data['file'] +" Player : "
    beginnings = ["I would","I want","I","Can I"]#
    questions = ["What if","Why not","What if we", "Can we","Maybe we","If I","Do I","Can I ","I ","What I'll do is I'll ", "Would it work if I ", "Can I cast ","I would like to cast ", "Can I roll a","Remind me about ","What about ","Where","How","What check","What","From where I am "] 
    beginning=random.choice(beginnings)
    if random.random() < 0.11:
        beginning = random.choice(questions)
    prediction = learn.predict(text+beginning, n_words=200)#learn.beam_search("How do you want to do this? Player :", n_words=100)
    result = prediction[len(text):]
    gm = result.lower().find("gm :",2)
    player = result.lower().find("player :",2)
    if player > -1:
        if gm > -1:
            if player < gm:
                result = result[:player]
            else:
                result = result[:gm]
        else:
            result = result[:player]
    else:
        if gm > -1:
            result = result[:gm]

    match = re.search(r'\d+', result)
    while match!=None:
        integer = match[0]
        place = result.find(str(integer))
        result = result[:place]
        rolls =["natural 1 ","1 ","2 ","3 ","4 ","5 ","6 ","7 ","8 ","9 ","10 ","11 ","12 ","13 ","14 ","15 ","16 ","17 ","18 ","19 ","natural 20 ","critical "]
        result = result+random.choice(rolls)
        prediction = learn.predict(text+result, n_words=200)#learn.beam_search("How do you want to do this? Player :", n_words=100)
        result = prediction[len(text):]
        gm = result.lower().find("gm :",2)
        player = result.lower().find("player :",2)
        if player > -1:
            if gm > -1:
                if player < gm:
                    result = result[:player]
                else:
                    result = result[:gm]
            else:
                result = result[:player]
        else:
            if gm > -1:
                result = result[:gm]

        match = re.search(r'\d+', result)
    return JSONResponse({'result': str(result)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
