# Hotel Booking Cancellation Prediction

Predicts whether a hotel guest will cancel a booking, using the booking details.

## Dataset
Hotel Booking Demand dataset (Kaggle): 119,390 bookings, 32 columns. About 37% of the bookings were cancelled.

## What I did
1. *Cleaning:* dropped company (94% missing), filled missing agent and children with 0, and missing country with "Unknown".
2. *Data leakage:* removed reservation_status, reservation_status_date and assigned_room_type, because they reveal whether the booking was cancelled and would give fake accuracy.
3. *Encoding:* kept the top 10 countries and grouped the rest as "Other", then one-hot encoded the text columns (69 features).
4. *Split:* 80/20 train-test split, stratified on the target.
5. *Models:* Logistic Regression (baseline, scaled data), Random Forest and XGBoost.

## EDA findings
- Cancelled bookings were made about 145 days ahead on average, against about 80 days for bookings that were kept.
- City Hotel cancellation rate is 41.7%, Resort Hotel is 27.8%.
- 99.4% of "Non Refund" bookings were cancelled, which looks odd and is probably a quirk of the dataset.

## Results (test set, 23,878 bookings)
| Model | Accuracy | Recall (cancel) | F1 (cancel) |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.64 | 0.72 |
| Random Forest | 0.90 | 0.83 | 0.86 |
| XGBoost | 0.88 | 0.81 | 0.84 |

Random Forest was the best model. It catches 83% of the bookings that actually get cancelled.

## Top features (Random Forest)
lead_time, deposit_type_Non Refund, country_PRT, adr, total_of_special_requests

## Limitations and next steps
- XGBoost was not tuned. Tuning both models with cross-validation may improve the results.
- agent is an ID but was treated as a number. It should be encoded as a category or dropped.
- Planned: a Streamlit app to try the model.

## Tools
Python, Pandas, NumPy, scikit-learn, XGBoost, Jupyter Notebook
