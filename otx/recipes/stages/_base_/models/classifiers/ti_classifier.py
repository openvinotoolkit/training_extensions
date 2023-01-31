_base_ = "./classifier.py"

model = dict(type="TaskIncrementalLwF", head=dict(type="TaskIncLwfHead"))
