from pydantic import BaseModel, Field

class EmployeeInferenceBase(BaseModel):
    Age: float = Field(..., description="Age in years")
    DailyRate: float = Field(..., description="Daily Rate")
    DistanceFromHome: float = Field(..., description="Distance from home in km")
    HourlyRate: float = Field(..., description="Hourly Rate")
    MonthlyIncome: float = Field(..., description="Monthly Income")
    MonthlyRate: float = Field(..., description="Monthly Rate")
    NumCompaniesWorked: float = Field(..., description="Number of companies worked at")
    PercentSalaryHike: float = Field(..., description="Recent percentage salary hike")
    TotalWorkingYears: float = Field(..., description="Total working years")
    TrainingTimesLastYear: float = Field(..., description="Training times last year")
    YearsAtCompany: float = Field(..., description="Years at company")
    YearsInCurrentRole: float = Field(..., description="Years in current role")
    YearsSinceLastPromotion: float = Field(..., description="Years since last promotion")
    YearsWithCurrManager: float = Field(..., description="Years with current manager")
    Education: float = Field(..., description="Education Level (1-5)")
    EnvironmentSatisfaction: float = Field(..., description="Environment Satisfaction (1-4)")
    JobInvolvement: float = Field(..., description="Job Involvement (1-4)")
    JobLevel: float = Field(..., description="Job Level (1-5)")
    JobSatisfaction: float = Field(..., description="Job Satisfaction (1-4)")
    PerformanceRating: float = Field(..., description="Performance Rating (1-4)")
    RelationshipSatisfaction: float = Field(..., description="Relationship Satisfaction (1-4)")
    StockOptionLevel: float = Field(..., description="Stock Option Level (0-3)")
    WorkLifeBalance: float = Field(..., description="Work Life Balance (1-4)")
    BusinessTravel: str = Field(..., description="Non-Travel, Travel_Rarely, Travel_Frequently")
    Department: str = Field(..., description="Sales, Research & Development, Human Resources")
    EducationField: str = Field(..., description="Life Sciences, Medical, Marketing, Technical Degree, HR, Other")
    Gender: str = Field(..., description="Male, Female")
    JobRole: str = Field(..., description="Specific job role (e.g., Sales Executive)")
    MaritalStatus: str = Field(..., description="Single, Married, Divorced")
    OverTime: str = Field(..., description="Yes, No")

class PredictionResponse(BaseModel):
    risk_score: float
    risk_tier: str
    optimal_threshold_used: float
    top_factors: list[dict]
