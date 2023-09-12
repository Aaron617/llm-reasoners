import dataclasses
import json
from typing import Optional, Union

import requests

from reasoners.algorithm import MCTSResult, BeamSearchResult
from reasoners.visualization import TreeLog, TreeLogEncoder

_API_DEFAULT_BASE_URL = "https://2wz3t0av30.execute-api.us-west-1.amazonaws.com/staging"
_VISUALIZER_DEFAULT_BASE_URL = "https://www.llm-reasoners.net"


class VisualizerClient:
    def __init__(self, base_url: str = _API_DEFAULT_BASE_URL) -> None:
        self.base_url = base_url

    @dataclasses.dataclass
    class TreeLogReceipt:
        id: str
        access_key: str

        @property
        def access_url(self) -> str:
            return f"{_VISUALIZER_DEFAULT_BASE_URL}/visualizer/{self.id}?accessKey={self.access_key}"

    def post_log(self, data: Union[TreeLog, str, dict]) -> Optional[TreeLogReceipt]:
        if isinstance(data, TreeLog):
            data = json.dumps(data, cls=TreeLogEncoder)
        if isinstance(data, dict):
            data = json.dumps(data)

        url = f"{self.base_url}/logs"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=data)

        if response.status_code != 200:
            print(f"POST Log failed with status code: {response.status_code}, message: {response.text}")
            return None

        return self.TreeLogReceipt(**response.json())


def present_visualizer(receipt: VisualizerClient.TreeLogReceipt):
    import webbrowser
    print(f"Visualizer URL: {receipt.access_url}")
    webbrowser.open(receipt.access_url)


def visualize(result: Union[TreeLog, MCTSResult, BeamSearchResult], **kwargs):
    tree_log: TreeLog
    print("kkkk")
    if isinstance(result, TreeLog):
        print(0)
        tree_log = result
    elif isinstance(result, MCTSResult):
        print(1)
        tree_log = TreeLog.from_mcts_results(result, **kwargs)
    elif isinstance(result, BeamSearchResult):
        print(2)
        tree_log = TreeLog.from_beam_search_results(result, **kwargs)
    elif isinstance(result, ...):
        raise NotImplementedError()
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")

    receipt = VisualizerClient().post_log(tree_log)

    if receipt is not None:
        present_visualizer(receipt)
