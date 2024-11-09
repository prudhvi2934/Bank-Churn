from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from churn.constants import APP_HOST, APP_PORT
from churn.pipline.prediction_pipeline import ChurnData, ChurnClassifier
from churn.pipline.training_pipeline import TrainPipeline


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Customer_Age: Optional[int] = None
        self.Gender: Optional[str] = None
        self.Dependent_count: Optional[int] = None
        self.Education_Level: Optional[str] = None
        self.Marital_Status: Optional[str] = None
        self.Income_Category: Optional[str] = None
        self.Card_Category: Optional[str] = None
        self.Months_on_book: Optional[int] = None
        self.Total_Relationship_Count: Optional[int] = None
        self.Months_Inactive_12_mon: Optional[int] = None
        self.Contacts_Count_12_mon: Optional[int] = None
        self.Credit_Limit: Optional[float] = None
        self.Total_Revolving_Bal: Optional[int] = None
        self.Avg_Open_To_Buy: Optional[float] = None
        self.Total_Amt_Chng_Q4_Q1: Optional[float] = None
        self.Total_Trans_Amt: Optional[int] = None
        self.Total_Trans_Ct: Optional[int] = None
        self.Total_Ct_Chng_Q4_Q1: Optional[float] = None
        self.Avg_Utilization_Ratio: Optional[float] = None

    async def get_churn_data(self):
        form = await self.request.form()
        self.Customer_Age = form.get("Customer_Age")
        self.Gender = form.get("Gender")
        self.Dependent_count = form.get("Dependent_count")
        self.Education_Level = form.get("Education_Level")
        self.Marital_Status = form.get("Marital_Status")
        self.Income_Category = form.get("Income_Category")
        self.Card_Category = form.get("Card_Category")
        self.Months_on_book = form.get("Months_on_book")
        self.Total_Relationship_Count = form.get("Total_Relationship_Count")
        self.Months_Inactive_12_mon = form.get("Months_Inactive_12_mon")
        self.Contacts_Count_12_mon = form.get("Contacts_Count_12_mon")
        self.Credit_Limit = form.get("Credit_Limit")
        self.Total_Revolving_Bal = form.get("Total_Revolving_Bal")
        self.Avg_Open_To_Buy = form.get("Avg_Open_To_Buy")
        self.Total_Amt_Chng_Q4_Q1 = form.get("Total_Amt_Chng_Q4_Q1")
        self.Total_Trans_Amt = form.get("Total_Trans_Amt")
        self.Total_Trans_Ct = form.get("Total_Trans_Ct")
        self.Total_Ct_Chng_Q4_Q1 = form.get("Total_Ct_Chng_Q4_Q1")
        self.Avg_Utilization_Ratio = form.get("Avg_Utilization_Ratio")


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
        "churn.html", {"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_churn_data()

        churn_data = ChurnData(
            Customer_Age=form.Customer_Age,
            Gender=form.Gender,
            Dependent_count=form.Dependent_count,
            Education_Level=form.Education_Level,
            Marital_Status=form.Marital_Status,
            Income_Category=form.Income_Category,
            Card_Category=form.Card_Category,
            Months_on_book=form.Months_on_book,
            Total_Relationship_Count=form.Total_Relationship_Count,
            Months_Inactive_12_mon=form.Months_Inactive_12_mon,
            Contacts_Count_12_mon=form.Contacts_Count_12_mon,
            Credit_Limit=form.Credit_Limit,
            Total_Revolving_Bal=form.Total_Revolving_Bal,
            Avg_Open_To_Buy=form.Avg_Open_To_Buy,
            Total_Amt_Chng_Q4_Q1=form.Total_Amt_Chng_Q4_Q1,
            Total_Trans_Amt=form.Total_Trans_Amt,
            Total_Trans_Ct=form.Total_Trans_Ct,
            Total_Ct_Chng_Q4_Q1=form.Total_Ct_Chng_Q4_Q1,
            Avg_Utilization_Ratio=form.Avg_Utilization_Ratio
        )

        churn_df = churn_data.get_churn_input_data_frame()

        model_predictor = ChurnClassifier()

        value = model_predictor.predict(dataframe=churn_df)[0]

        status = None
        if value == 1:
            status = "Attrited Customerr"
        else:
            status = "Existing Customer"

        return templates.TemplateResponse(
            "churn.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
