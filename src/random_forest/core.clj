(ns random-forest.core
  (:require [clojure.java.io :as io]
           [clojure.data.csv :as csv])
  (:import weka.core.converters.ConverterUtils$DataSource
           weka.core.Instances
           weka.core.Instance
           weka.core.Attribute
           weka.classifiers.Classifier
           weka.classifiers.trees.RandomForest))

(set! *warn-on-reflection* true)

(defonce TRAINING-FILE "data/train.csv")
(defonce TEST-FILE "data/test.csv")
(defonce NUMCLASS {"Zero"     0
                   "One"      1
                   "Two"      2
                   "Three"    3
                   "Four"     4
                   "Five"     5
                   "Six"      6
                   "Seven"    7
                   "Eight"    8
                   "Nine"     9})
(def class-attr (atom nil))
(def results (atom []))

(defn train-instances []
  (let [^ConverterUtils$DataSource ds (ConverterUtils$DataSource. TRAINING-FILE)
        ^Instances instances (.getDataSet ds 0)]
    (reset! class-attr (.classAttribute instances))
    instances))

(defn test-instances ^Instances []
  (let [^ConverterUtils$DataSource ds (ConverterUtils$DataSource. TEST-FILE)
        ^Instances instances (.getDataSet ds)]
    instances))

(defn train-classifier [^Instances instances]
  (let [^RandomForest rf (RandomForest.)]
    (.setClassIndex instances 0)
    (.setNumTrees rf 100)
    (.setNumFeatures rf 28)
    (.buildClassifier rf instances)
    rf))

(defn insert-class-attr [^Instances instances]
  (let [new-attr (.copy ^Attribute @class-attr "label")]
    (.insertAttributeAt instances new-attr 0)
    (.setClassIndex instances 0)
    instances))

(defn predict [^Classifier model ^Instance instance]
  (let [
        instance-label (.classifyInstance model instance)
        label-val (.value ^Attribute @class-attr (int instance-label))]
    (reset! results (conj @results (NUMCLASS label-val)))))

(defn write-results []
  (let [writer (io/writer "data/results.csv")]
    (csv/write-csv writer [["ImageId" "Label"]])
    (doseq [i (range (count @results))]
      (csv/write-csv writer [[(+ 1 i) (nth @results i)]]))
    (.close writer)))

(defn -main []
  (let [model (train-classifier (train-instances))
        ^Instances tests (insert-class-attr (test-instances))]
    (dorun (map #(predict model %) tests))
    (write-results)))
