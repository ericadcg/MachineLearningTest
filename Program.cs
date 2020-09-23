using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Microsoft.ML.FastTree;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.DataOperationsCatalog;


namespace ML_Test
{
    class TrainingData
{
        [LoadColumn(0)]
        public string message { get; set; }

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool success { get; set; }

    }

    class DataPrediction : TrainingData
    {   
        public float Score { get; set; }

        public float Probability { get; set; }

       
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }



    class Program
    {
        static List<TrainingData> trainingData = new List<TrainingData>();

        static void LoadTrainingData()
        {
            //Data that will be used to train ML
            trainingData.Add(new TrainingData() { message = "this is good", success = true });
            trainingData.Add(new TrainingData() { message = "this is bad", success = false });
            trainingData.Add(new TrainingData() { message = "yes yes", success = true });
            trainingData.Add(new TrainingData() { message = "no no", success = true });
            trainingData.Add(new TrainingData() { message = "wew", success = false });
            trainingData.Add(new TrainingData() { message = "yeaaah", success = true });
            trainingData.Add(new TrainingData() { message = "horrible", success = false });
            trainingData.Add(new TrainingData() { message = "well yes", success = true });
            trainingData.Add(new TrainingData() { message = "perfect", success = true });
            trainingData.Add(new TrainingData() { message = "oh no", success = false });
            trainingData.Add(new TrainingData() { message = "oh nice", success = true });
            trainingData.Add(new TrainingData() { message = "ups", success = false });
            trainingData.Add(new TrainingData() { message = "good, perfect", success = true });
            trainingData.Add(new TrainingData() { message = "oh terrible", success = false });
            trainingData.Add(new TrainingData() { message = "so good, so perfect", success = true });
            trainingData.Add(new TrainingData() { message = "bad, terrible", success = false });
            trainingData.Add(new TrainingData() { message = "yeah good", success = true });
            trainingData.Add(new TrainingData() { message = "so awful, terrible", success = false });
            trainingData.Add(new TrainingData() { message = "i love this", success = true });
            trainingData.Add(new TrainingData() { message = "Can I have some more?", success = true });
            trainingData.Add(new TrainingData() { message = "great job", success = true });
            trainingData.Add(new TrainingData() { message = "amazing app", success = true });
            trainingData.Add(new TrainingData() { message = "very nice", success = true });
            trainingData.Add(new TrainingData() { message = "i love it", success = true });
            trainingData.Add(new TrainingData() { message = "i hate it", success = false });
            trainingData.Add(new TrainingData() { message = "no more", success = false });
            trainingData.Add(new TrainingData() { message = "could be better", success = false });
            trainingData.Add(new TrainingData() { message = "couldn't be better", success = true });
            trainingData.Add(new TrainingData() { message = "could not be better", success = true });
            trainingData.Add(new TrainingData() { message = "can you improve it?", success = false });
            trainingData.Add(new TrainingData() { message = "awful", success = false });
            trainingData.Add(new TrainingData() { message = "not very good", success = false });
            trainingData.Add(new TrainingData() { message = "very good", success = true });
            trainingData.Add(new TrainingData() { message = "very bad", success = false });

        }

        static List<TrainingData> testData = new List<TrainingData>();

        static void LoadTestData()
        {
            //Data that will be used to TEST ML. This data will gives us a good evaluation for the model as it's very simple and direct. Training data is more important to have a good model.
            testData.Add(new TrainingData() { message = "good", success = true });
            testData.Add(new TrainingData() { message = "bad", success = false });
            testData.Add(new TrainingData() { message = "yes", success = true });
            testData.Add(new TrainingData() { message = "yeah", success = true });
            testData.Add(new TrainingData() { message = "no", success = false });
            testData.Add(new TrainingData() { message = "nice", success = true });
            testData.Add(new TrainingData() { message = "horrible", success = false });
            testData.Add(new TrainingData() { message = "awesome", success = true });
            testData.Add(new TrainingData() { message = "awful", success = false });
            testData.Add(new TrainingData() { message = "wow", success = true });
            testData.Add(new TrainingData() { message = "terrible", success = false });
            testData.Add(new TrainingData() { message = "amazing", success = true });
            testData.Add(new TrainingData() { message = "not good", success = false });
        }


        static void Main(string[] args)
        {
            //Loads training data
            LoadTrainingData();

            var mlContext = new MLContext();

            //Convert training data in DataView
            IDataView dataView = mlContext.Data.LoadFromEnumerable<TrainingData>(trainingData);

            ITransformer model2 = BuildAndTrainModel(mlContext, dataView);
            
            //Loads test data
            LoadTestData();

            //Convert test data in DataView
            IDataView dataViewtest = mlContext.Data.LoadFromEnumerable<TrainingData>(testData);

            Evaluate(mlContext, model2, dataViewtest);

            //ML is ready to "guess" message sentiment

            Console.WriteLine();
            Console.WriteLine("Please input a simple message to be evaluated.");
            string inpMsg = Console.ReadLine();
            GetMessagePrediction(mlContext, model2, inpMsg);
        }

        // Support functions
        //Builds and trains ML Model with training data from above
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView TrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText("Features", "message")
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("===== Please wait while the ML model is being created and trained. =====");
            var model = estimator.Fit(TrainSet);
            Console.WriteLine("===== End of training =====");
            Console.WriteLine();

            return model;
        }

        // Gets ML evaluation results from testing with above data
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView TestSet)
        {
            
            IDataView predictions = model.Transform(TestSet);

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model evaluation:");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine();

        }

        //Prediction of a single message
        private static void GetMessagePrediction(MLContext mlContext, ITransformer model, string inputMessage)
        {

            PredictionEngine<TrainingData, DataPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<TrainingData, DataPrediction>(model);

            TrainingData sampleStatement = new TrainingData
            {
                message = inputMessage
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            float probPercentage = (float)Math.Round(resultPrediction.Probability * 100, 2);
            
            Console.WriteLine();
            Console.WriteLine($" Your message: {resultPrediction.message} \n Sentiment Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} \n Probability: {probPercentage}%");
            Console.WriteLine();
        }

    }
}
