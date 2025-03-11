from methodology_composing_using_template import methodology_composing
from related_work_composing_using_template import related_work_composing
from experiments_composing import experiments_composing
from introduction_composing import introduction_composing
from conclusion_composing import conclusion_composing
from abstract_composing import abstract_composing
import asyncio
import argparse

async def writing(research_field: str, instance_id: str):
    await methodology_composing(research_field, instance_id)
    await related_work_composing(research_field, instance_id)
    await experiments_composing(research_field, instance_id)
    await introduction_composing(research_field, instance_id)
    await conclusion_composing(research_field, instance_id)
    await abstract_composing(research_field, instance_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, default="vq")
    parser.add_argument("--instance_id", type=str, default="rotation_vq")
    args = parser.parse_args()
    asyncio.run(writing(args.research_field, args.instance_id))

        