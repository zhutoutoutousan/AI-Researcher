import dataclasses
from tree_sitter import Language
import tree_sitter
import glob
import uuid
from loguru import logger

@dataclasses.dataclass
class Snippet:
    """Dataclass for storing Embedded Snippets"""

    id: str
    embedding: list[float] | None
    snippet: str
    filename: str
    language: str


class CodeParser:
    """Code Parser Class."""

    def __init__(self, language: str, node_types: list[str], path_to_object_file: str):
        self.node_types = node_types
        self.language = language
        try:
            self.parser = tree_sitter.Parser()
            self.parser.set_language(
                tree_sitter.Language(f"{path_to_object_file}/my-languages.so", language)
            )
        except Exception as e:
            logger.exception("failed to build %s parser: ", e)

    def parse_file(self, content: str, filename: str):
        """
        Parse code snippets from single code file.

        Args:
            content: The content of the file.
            filename: The name of the code file.

        Returns:
        List of Parsed Snippets
        """
        try:
            tree = self.parser.parse(content)
        except Exception as e:
            logger.error(f"Failed to parse snippet: {filename} \n Error: {e}")
            return

        cursor = tree.walk()
        parsed_snippets = []

        # Walking nodes from abstract syntax tree
        while cursor.goto_first_child():
            if cursor.node.type in self.node_types:
                parsed_snippets.append(
                    Snippet(
                        id=str(uuid.uuid4()),
                        snippet=cursor.node.text,
                        filename=filename,
                        language=self.language,
                        embedding=None,
                    )
                )

            while cursor.goto_next_sibling():
                if cursor.node.type in self.node_types:
                    parsed_snippets.append(
                        Snippet(
                            id=str(uuid.uuid4()),
                            snippet=cursor.node.text,
                            filename=filename,
                            language=self.language,
                            embedding=None,
                        )
                    )
        return parsed_snippets

    def parse_directory(self, code_directory_path):
        """
        Parse code snippets from all files in directory.

        Args:
            code_directory_path: Directory path containing code files.

        Returns:
        List of Parsed Snippets
        """
        parsed_contents = []
        for filename in glob.glob(f"{code_directory_path}/**/*.py", recursive=True):
            # print(filename)
            with open(filename, "rb") as codefile:
                code_content = codefile.read()

            parsed_content = self.parse_file(code_content, filename)
            parsed_contents.extend(parsed_content)

        return parsed_contents
    
def to_dataframe_row(embedded_snippets: list[Snippet]):
    """
    Helper function to convert Embedded Snippet object to a dataframe row
    in dictionary format.

    Args:
        embedded_snippets: List of Snippets to be converted

    Returns:
        List of Dictionaries
    """
    outputs = []
    for embedded_snippet in embedded_snippets:
        output = {
            "ids": embedded_snippet.id,
            "embeddings": embedded_snippet.embedding,
            "snippets": embedded_snippet.snippet,
            "metadatas": {
                "filenames": embedded_snippet.filename,
                "languages": embedded_snippet.language,
            },
        }
        outputs.append(output)
    return outputs