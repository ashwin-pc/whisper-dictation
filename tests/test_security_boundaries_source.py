import ast
import os
import pathlib
import unittest


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(
    os.environ.get("TARGET_REPOSITORY_ROOT", DEFAULT_ROOT)
)


class SecurityBoundarySourceTests(unittest.TestCase):
    def test_every_synthetic_typing_sink_uses_prepared_safe_text(self):
        sinks = []
        for relative_path in ("src/main.py", "src/text_selection.py"):
            source = (REPOSITORY_ROOT / relative_path).read_text()
            tree = ast.parse(source, filename=relative_path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "type"
                ):
                    sinks.append((relative_path, node.lineno, node.args))

        self.assertEqual(len(sinks), 3)
        for relative_path, line_number, arguments in sinks:
            with self.subTest(file=relative_path, line=line_number):
                self.assertEqual(len(arguments), 1)
                self.assertIsInstance(arguments[0], ast.Name)
                self.assertEqual(arguments[0].id, "safe_text")

    def test_bedrock_request_uses_the_structured_request_builder(self):
        source = (REPOSITORY_ROOT / "src/bedrock_client.py").read_text()
        tree = ast.parse(source, filename="src/bedrock_client.py")

        builder_calls = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_bedrock_request"
            )
        ]

        self.assertEqual(len(builder_calls), 1)


if __name__ == "__main__":
    unittest.main()
