// GlucoseStore.swift
// Reads CGM data from HealthKit and runs Core ML inference.
// Mirrors the data layer behind all 4 GlucoSense watch screens.

import Foundation
import HealthKit
import CoreML
import Combine

@MainActor
class GlucoseStore: ObservableObject {

    // MARK: - Published state (drives all 4 screens)
    @Published var currentGlucose: Double = 0          // mg/dL
    @Published var rateOfChange: Double = 0            // mg/dL per 5 min
    @Published var recentReadings: [GlucoseReading] = []
    @Published var prediction30min: Double?
    @Published var prediction60min: Double?
    @Published var authorizationStatus: HKAuthorizationStatus = .notDetermined

    // MARK: - Private
    private let healthStore = HKHealthStore()
    private var model: GlucoSense_lstm?               // Core ML model

    // Normalisation constants — must match training scaler (MinMaxScaler, range 40–400)
    private let glucoseMin: Double = 40
    private let glucoseMax: Double = 400

    init() {
        loadModel()
        requestAuthorization()
    }

    // MARK: - Core ML

    private func loadModel() {
        // Add GlucoSense_lstm.mlpackage to the Xcode target before building.
        model = try? GlucoSense_lstm(configuration: MLModelConfiguration())
    }

    // MARK: - HealthKit authorization

    private func requestAuthorization() {
        guard HKHealthStore.isHealthDataAvailable() else { return }

        let glucoseType = HKQuantityType(.bloodGlucose)
        healthStore.requestAuthorization(toShare: [], read: [glucoseType]) { [weak self] success, _ in
            Task { @MainActor in
                if success {
                    self?.authorizationStatus = .sharingAuthorized
                    self?.startObserving()
                    self?.fetchHistory()
                }
            }
        }
    }

    // MARK: - Fetch last 2 hours of CGM readings (input window for model)

    func fetchHistory() {
        let glucoseType = HKQuantityType(.bloodGlucose)
        let now = Date()
        let twoHoursAgo = now.addingTimeInterval(-2 * 3600)
        let predicate = HKQuery.predicateForSamples(withStart: twoHoursAgo, end: now)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)

        let query = HKSampleQuery(
            sampleType: glucoseType,
            predicate: predicate,
            limit: 24,
            sortDescriptors: [sort]
        ) { [weak self] _, samples, _ in
            guard let self, let samples = samples as? [HKQuantitySample] else { return }

            Task { @MainActor in
                self.recentReadings = samples.map {
                    GlucoseReading(
                        timestamp: $0.startDate,
                        value: $0.quantity.doubleValue(for: .init(from: "mg/dL"))
                    )
                }
                self.currentGlucose = self.recentReadings.last?.value ?? 0
                if self.recentReadings.count >= 2 {
                    self.rateOfChange = self.recentReadings.last!.value
                                      - self.recentReadings[self.recentReadings.count - 2].value
                }
                self.runInference()
            }
        }
        healthStore.execute(query)
    }

    // MARK: - Live observer (fires on every new CGM reading)

    private func startObserving() {
        let glucoseType = HKQuantityType(.bloodGlucose)
        let query = HKObserverQuery(sampleType: glucoseType, predicate: nil) { [weak self] _, _, error in
            guard error == nil else { return }
            Task { @MainActor in self?.fetchHistory() }
        }
        healthStore.execute(query)

        // Enable background delivery so the app wakes even when the watch face is off
        healthStore.enableBackgroundDelivery(for: glucoseType, frequency: .immediate) { _, _ in }
    }

    // MARK: - Inference

    private func runInference() {
        guard let model, recentReadings.count == 24 else { return }

        // Normalise to [0, 1] — same transform used during training
        let normalised = recentReadings.map { reading -> Float in
            Float((reading.value - glucoseMin) / (glucoseMax - glucoseMin))
        }

        // Build MLMultiArray of shape (1, 24, 1)
        guard let input = try? MLMultiArray(shape: [1, 24, 1], dataType: .float32) else { return }
        for i in 0..<24 {
            input[[0, i, 0] as [NSNumber]] = NSNumber(value: normalised[i])
        }

        guard let output = try? model.prediction(cgm_window: input) else { return }

        // Inverse normalise back to mg/dL
        let pred30 = Double(truncating: output.glucose_predictions[[0, 0] as [NSNumber]])
        let pred60 = Double(truncating: output.glucose_predictions[[0, 1] as [NSNumber]])

        prediction30min = pred30 * (glucoseMax - glucoseMin) + glucoseMin
        prediction60min = pred60 * (glucoseMax - glucoseMin) + glucoseMin
    }
}

// MARK: - Data types

struct GlucoseReading: Identifiable {
    let id = UUID()
    let timestamp: Date
    let value: Double   // mg/dL
}
