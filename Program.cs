// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using ProductSalesAnomalyDetection;

namespace ProductSalesAnomalyDetection
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        //assign the Number of records in dataset file to constant variable
        const int _docsize = 36;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Load the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            DetectSpike(mlContext, _docsize, dataView);

            DetectChangepoint(mlContext, _docsize, dataView);
        }

        //Add the CreateEmptyDataView() method
        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Created empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        // Created the DetectSpike() method
        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95d, pvalueHistoryLength: docSize / 4);

            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

            IDataView transformedData = iidSpikeTransform.Transform(productSales);

            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value");

            foreach (var p in predictions)
            {
                if (p.Prediction is not null)
                {
                    var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                    if (p.Prediction[0] == 1)
                    {
                        results += " <-- Spike detected";
                    }

                    Console.WriteLine(results);
                }
            }
            Console.WriteLine("");

        }

        // Created the DetectChangepoint() method
        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95d, changeHistoryLength: docSize / 4);

            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));

            IDataView transformedData = iidChangePointTransform.Transform(productSales);

            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

            foreach (var p in predictions)
            {
                if (p.Prediction is not null)
                {
                    var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                    if (p.Prediction[0] == 1)
                    {
                        results += " <-- alert is on, predicted changepoint";
                    }
                    Console.WriteLine(results);
                }
            }
            Console.WriteLine("");

        }
    }
}

