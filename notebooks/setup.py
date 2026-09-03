import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer, mo, torch


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, torch):
    repo = "meta-llama/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to("mps")
    return model, tok


@app.cell
def _(model, tok):
    ids = tok("The present King of France is", return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=16, do_sample=False)
    print(tok.decode(out[0]))
    return (out,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    for now, at least, Russell lives
    """)
    return


@app.cell
def _(model):
    model.state_dict()
    return


@app.cell
def _(out):
    out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
