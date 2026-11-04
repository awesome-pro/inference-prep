import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch(session, url) for url in urls
        ]

        results = await asyncio.gather(*tasks)

asyncio.run(main())