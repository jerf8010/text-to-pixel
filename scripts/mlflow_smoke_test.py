import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello"):
    mlflow.log_param("lr", 3e-4)
    for step in range(50):
        mlflow.log_metric("fake_loss", math.exp(-step / 10), step=step)

    fig, ax = plt.subplots()
    ax.plot([math.exp(-s / 10) for s in range(50)])
    fig.savefig("fake_curve.png")
    mlflow.log_artifact("fake_curve.png")

print("done")