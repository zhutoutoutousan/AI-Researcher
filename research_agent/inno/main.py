from research_agent.inno import MetaChain, Agent, Response
from typing import List
from research_agent.inno.logger import MetaChainLogger
async def run_in_client(
    agent: Agent,
    messages: List,
    context_variables: dict = {},
    logger: MetaChainLogger = None,
):
    """
    """
    client = MetaChain(log_path=logger)

    MAX_RETRY = 3
    for i in range(MAX_RETRY):
        try:
            response: Response = await client.run_async(agent, messages, context_variables, debug=True)
        except Exception as e:
            logger.info(f'Exception in main loop: {e}', title='ERROR', color='red')
            raise e
        if 'Case resolved' in response.messages[-1]['content']:
            break
        elif 'Case not resolved' in response.messages[-1]['content']:
            messages.extend(response.messages)
            messages.append({
                'role': 'user',
                'content': 'Please try to resolve the case again. It\'s important for me to resolve the case. Trying again in another way may be helpful.'
            })

    return response