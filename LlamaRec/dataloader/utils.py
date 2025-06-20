import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # template_name = "alpaca"
            template_name = "alpaca_short"
        file_name = osp.join("dataloader", "templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


# Neo4j queries
QUERIES = {"ml-100k" : {'q_u_h' : """MATCH (u:User {id: $u_id}), (m:Movie {id: $i_id})
                                    MATCH p = shortestPath((u)-[*]->(m))
                                    RETURN [n in nodes(p) | n.id], [n in Relationships(p) | type(n)], [n in Relationships(p) | n.rating]""",
                        'q_h_c' : """MATCH (m1:Movie {id: $h_id}), (m2:Movie {id: $i_id})
                                    MATCH p = shortestPath((m1)-[*]->(m2))
                                    RETURN [n in nodes(p) | n.id], [n in Relationships(p) | type(n)]"""},
           
           'beauty': {'q_u_h': """MATCH (u:User {id: $u_id}), (i:Item {id: $i_id})
                                 MATCH p = shortestPath((u)-[*]->(i))
                                 RETURN [n in nodes(p) | n.id], [n in Relationships(p) | type(n)], [n in Relationships(p) | n.rating]""",
                      'q_h_c' : """MATCH (i1:Item {id: $h_id}), (i2:Item {id: $i_id})
                                  MATCH p = shortestPath((i1)-[*]->(i2))
                                  RETURN [n in nodes(p) | n.id], [n in Relationships(p) | type(n)]"""}
          }