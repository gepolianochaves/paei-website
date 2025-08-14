# Classification by Logistic Regression (Python)

## Configuration Chunks for Python Part



There was an error at the end regarding matplotlib. One reference link mentioned installing matplotlib before pandas.


``` bash
pip install matplotlib
```

Install pandas in this environment


``` bash
pip install pandas
```

## Shiny Application


``` r
# Data pre-processing ----
# Tweak the "am" variable to have nicer factor labels -- since this
# doesn't rely on any user inputs, we can do this once at startup
# and then use the value throughout the lifetime of the app

library(shiny)
library(dplyr)
library(ggplot2)

r2_gse62564_GSVA_Metadata <- readRDS( "~/Desktop/Gepoliano/Bioinformatics Bridge Course/2023-bioinfo-bridge-course/recombio/6-Neuroblastoma/results/r2_gse62564_GSVA_Metadata.rds")

mpgData <- mtcars
mpgData$am <- factor(mpgData$am, labels = c("Automatic", "Manual"))

r2_gse62564_GSVA_Metadata$MYCN <- as.numeric(r2_gse62564_GSVA_Metadata$MYCN)

r2_gse62564_GSVA_Metadata$high_risk <- factor(r2_gse62564_GSVA_Metadata$high_risk, labels = c("yes", "no"))
r2_gse62564_GSVA_Metadata$inss_stage <- factor(r2_gse62564_GSVA_Metadata$inss_stage, labels = c("st1", "st2", "st3", "st4", "st4s"))
r2_gse62564_GSVA_Metadata$mycn_status <- factor(r2_gse62564_GSVA_Metadata$mycn_status, labels = c("mycn_amp", "mycn_nonamp", "na"))

# Define server logic to plot various variables against mpg ----
server <- function(input, output) {
  
  # Compute the formula text ----
  # This is in a reactive expression since it is shared by the
  # output$caption and output$mpgPlot functions
  formulaText <- reactive({
    paste("MYCN ~", input$variable)
  })
  
  # Return the formula text for printing as a caption ----
  output$caption <- renderText({
    formulaText()
  })
  
  # Generate a plot of the requested variable against mpg ----
  # and only exclude outliers if requested
  output$mpgPlot <- renderPlot({
    boxplot(as.formula(formulaText()),
            data = r2_gse62564_GSVA_Metadata,
            outline = input$outliers,
            col = "#007bc2", pch = 19)
  })
  
}


# Define UI for miles per gallon app ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("Gene Expression Per Clinical Variable"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      
      # Input: Selector for variable to plot against mpg ----
      selectInput("variable", "Variable:",
                  c("High Risk" = "high_risk",
                    "INSS Stage" = "inss_stage",
                    "MYCN Status" = "mycn_status")),
      
      # Input: Checkbox for whether outliers should be included ----
      checkboxInput("outliers", "Show outliers", TRUE)
      
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Formatted text for caption ----
      h3(textOutput("caption")),
      
      # Output: Plot of the requested variable against mpg ----
      plotOutput("mpgPlot")
      
    )
  )
)


# View App
shinyApp(ui, server)
```

## Python Neuroblastoma Risk Classifier Machine


``` python
####################################################################
########## Part 1: Data Pre-processing; Get started on data analysis
####################################################################

import pandas as pd


dataset_kocak = pd.read_csv("data/r2_gse62564_GSVA_Metadata_selected.csv")
dataset_kocak.head(10)


X = dataset_kocak.iloc[:, 1:-1].values
y = dataset_kocak.iloc[:, -1]


########## Creating the Training Set and the Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

########## Feature Scaling the Feature Array

from sklearn.preprocessing import StandardScaler
# Standardization based on the features of the whole dataset (?)
# Compute in the training set (?)
# instance of the class
sc = StandardScaler()
# compute average and sd of the features
# Takes on the array of independent variables you want to scale
sc.fit_transform(X_train)
# We will only need the new array of independent variables in the training set
X_train = sc.fit_transform(X_train)

###################################################
########## Part 2 - Building and training the model
###################################################

########## Import library
from sklearn.linear_model import LogisticRegression


########## Building the model
model = LogisticRegression(random_state = 0 )


########## Training the model
model.fit(X_train, y_train)


########## Access coefficients for variable importance
coefficients = model.coef_[0]


########## Plot variable importance

X_test_kocak = dataset_kocak.drop('high_risk', axis=1)
X_test_kocak = X_test_kocak.drop('sample id', axis=1)


########## Plot variable importance

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

feature_importance = pd.DataFrame({'Feature': X_test_kocak.columns, 'Importance': np.abs(coefficients)})
# feature_importance = feature_importance.sort_values('Importance', ascending=True).head(70)
feature_importance = feature_importance.sort_values('Importance', ascending=True)
# feature_importance = feature_importance[:5000]
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

########## Inference

# Predictions for the test set and for a particular patient
y_pred = model.predict(sc.transform(X_test)) # First, call the scaler object

input_string = input('Enter elements of a list separated by space \n')
user_list = input_string.split()
print('User list: ', user_list)
model.predict(sc.transform([user_list]))
```

