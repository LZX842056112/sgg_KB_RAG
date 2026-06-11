from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="xxxx")


def do(name: str):
    pass


@app.get("/do/main")
async def do_main(background_task: BackgroundTasks):
    # 开启一个异步任务 [工作耗时 10s]
    # 怎么把函数 -> 扔到fastapi -> 开启的event_loop
    background_task.add_task(do, "二狗子")
    return {"code": 200}


async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流


@app.get("/stream")
async def stream_response():
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8001
    )
