from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# initialize Spark
print("Initializing Spark...")
spark = SparkSession.builder \
    .appName("CancerDiagnosis") \
    .master("local[*]") \
    .getOrCreate()

# load data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
df = spark.read.csv("project3_data.csv", header=True, inferSchema=True)
print(f"Loaded {df.count()} samples with {len(df.columns)} columns")

# show class distribution
print("\nClass Distribution:")
df.groupBy("diagnosis").count().show()

# convert label (B/M) to numeric (0/1)
print("\nConverting labels to numeric...")
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
df = indexer.fit(df).transform(df)

# prepare features, exclude 'id' and 'diagnosis' columns
feature_columns = [col for col in df.columns if col not in ["id", "diagnosis", "label"]]
print(f"\nUsing {len(feature_columns)} features:")
print(feature_columns[:5], "... (showing first 5)")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# select only what we need
df = df.select("features", "label")

# split data (80% train, 20% test)
print("\n" + "="*60)
print("SPLITTING DATA")
print("="*60)
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Training samples: {train_df.count()}")
print(f"Testing samples: {test_df.count()}")

# algorithm 1: random forest
print("\n" + "="*60)
print("ALGORITHM 1: RANDOM FOREST")
print("="*60)
print("Training Random Forest Classifier...")
print("Parameters: numTrees=100, maxDepth=10")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# evaluate random forest
print("\n--- Random Forest Results ---")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

rf_accuracy = evaluator_acc.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)
rf_precision = evaluator_precision.evaluate(rf_predictions)
rf_recall = evaluator_recall.evaluate(rf_predictions)

print(f"Accuracy:  {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")

# show some predictions
print("\nSample Predictions (first 10):")
rf_predictions.select("label", "prediction", "probability").show(10, truncate=False)

# algorithm 2: logistic regression
print("\n" + "="*60)
print("ALGORITHM 2: LOGISTIC REGRESSION")
print("="*60)
print("Training Logistic Regression Classifier...")
print("Parameters: maxIter=100, regParam=0.01")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)

# evaluate logistic regression
print("\n--- Logistic Regression Results ---")
lr_accuracy = evaluator_acc.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)
lr_precision = evaluator_precision.evaluate(lr_predictions)
lr_recall = evaluator_recall.evaluate(lr_predictions)

print(f"Accuracy:  {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"F1 Score:  {lr_f1:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")

# show some predictions
print("\nSample Predictions (first 10):")
lr_predictions.select("label", "prediction", "probability").show(10, truncate=False)

# comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"\n{'Metric':<15} {'Random Forest':<20} {'Logistic Regression':<20}")
print("-" * 55)
print(f"{'Accuracy':<15} {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)      {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"{'F1 Score':<15} {rf_f1:.4f}               {lr_f1:.4f}")
print(f"{'Precision':<15} {rf_precision:.4f}               {lr_precision:.4f}")
print(f"{'Recall':<15} {rf_recall:.4f}               {lr_recall:.4f}")

# determine winner
print("\n" + "="*60)
if rf_f1 > lr_f1:
    print(f"Random Forest performs better (F1: {rf_f1:.4f} vs {lr_f1:.4f})")
else:
    print(f"Logistic Regression performs better (F1: {lr_f1:.4f} vs {rf_f1:.4f})")
print("="*60)

spark.stop()
print("\nTraining complete!")