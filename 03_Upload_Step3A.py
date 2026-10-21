
# STEP 3 — Run concept-candidate analysis

from google.colab import files

uploaded = files.upload()

script = next(
    (name for name in uploaded if name.endswith(".py")),
    None
)

if script is None:
    raise ValueError("Upload the Step 3 Python file.")

with open(script, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, script, "exec"), globals())
