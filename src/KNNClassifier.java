import java.io.File;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk; // Thay thế J48 bằng IBk để sử dụng KNN
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class KNNClassifier {

    public static void main(String args[]) {

        int kFolds = 5;  // kFolds
        int k = 7;  // k

        try {
            // Tạo bộ phân loại KNN bằng cách tạo đối tượng của lớp IBk
            IBk knnClassifier = new IBk();
            knnClassifier.setKNN(k); // Đặt giá trị K=3

            // Đường dẫn đến tập dữ lSiệu
            String adultDataset = "./lib/adult.csv";

            // Tạo đối tượng CSVLoader để đọc tập dữ liệu CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(adultDataset));

            // Tạo các trường dữ liệu
            Instances datasetInstances = loader.getDataSet();

            // Đặt lớp mục tiêu
            datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

            // Đánh giá bằng cách tạo đối tượng của lớp Evaluation
            Evaluation evaluation = new Evaluation(datasetInstances);

            // Kiểm tra mô hình với k lần chia dữ liệu (k folds)
            evaluation.crossValidateModel(knnClassifier, datasetInstances, kFolds, new Random(1));

            // Hiển thị tổng kết
            System.out.println("=== Summary ===");
            System.out.printf("Correctly Classified Instances       %d               %.4f %%\n", (int) evaluation.correct(), evaluation.pctCorrect());
            System.out.printf("Incorrectly Classified Instances     %d               %.4f %%\n", (int) evaluation.incorrect(), 100 - evaluation.pctCorrect());
            System.out.printf("Kappa statistic                      %.4f\n", evaluation.kappa());
            System.out.printf("Mean absolute error                  %.4f\n", evaluation.meanAbsoluteError());
            System.out.printf("Root mean squared error              %.4f\n", evaluation.rootMeanSquaredError());
            System.out.printf("Relative absolute error              %.4f %%\n", evaluation.relativeAbsoluteError());
            System.out.printf("Root relative squared error          %.4f %%\n", evaluation.rootRelativeSquaredError());
            System.out.printf("Total Number of Instances            %d\n\n", datasetInstances.numInstances());

            // Hiển thị độ chính xác chi tiết theo lớp
            System.out.println("=== Detailed Accuracy By Class ===");
            System.out.println("\n                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class");

            double weightedFMeasure = 0.0;
            double weightedMCC = 0.0;
            double totalWeight = 0.0;

            for (int i = 0; i < datasetInstances.numClasses(); i++) {
                double weight = datasetInstances.attributeStats(datasetInstances.classIndex()).nominalCounts[i];
                double fMeasure = evaluation.fMeasure(i);
                double mcc = evaluation.matthewsCorrelationCoefficient(i);

                weightedFMeasure += weight * fMeasure;
                weightedMCC += weight * mcc;
                totalWeight += weight;

                System.out.printf("                 %.3f    %.3f    %.3f      %.3f    %.3f      %.3f    %.3f     %.3f     %s\n",
                        evaluation.truePositiveRate(i), evaluation.falsePositiveRate(i),
                        evaluation.precision(i), evaluation.recall(i),
                        fMeasure, mcc,
                        evaluation.areaUnderROC(i), evaluation.areaUnderPRC(i),
                        datasetInstances.classAttribute().value(i));
            }

            weightedFMeasure /= totalWeight;
            weightedMCC /= totalWeight;

            System.out.printf("Weighted Avg.    %.3f    %.3f    %.3f      %.3f    %.3f      %.3f    %.3f     %.3f\n\n",
                    evaluation.weightedTruePositiveRate(), evaluation.weightedFalsePositiveRate(),
                    evaluation.weightedPrecision(), evaluation.weightedRecall(),
                    weightedFMeasure, weightedMCC,
                    evaluation.weightedAreaUnderROC(), evaluation.weightedAreaUnderPRC());

            // Hiển thị ma trận nhầm lẫn
            System.out.println("=== Confusion Matrix ===");
            System.out.println("\n     a     b   <-- classified as");
            System.out.println(evaluation.toMatrixString());

        } catch (Exception e) {
            // In thông báo trên console
            System.out.println("Đã xảy ra lỗi!!!! \n" + e.getMessage());
        }
    }
}
