// AlertView.swift — Screen 4
// Nocturnal hypoglycemia smart alert — the most critical screen in the system.
// Fires when glucose is dropping toward the hypo threshold at night.

import SwiftUI
import WatchKit

struct AlertView: View {
    @EnvironmentObject var store: GlucoseStore
    @State private var acknowledged = false

    var isHypoglycemiaRisk: Bool {
        store.currentGlucose < 90 && store.rateOfChange < -1.0
    }

    var body: some View {
        if isHypoglycemiaRisk && !acknowledged {
            alertContent
        } else {
            safeContent
        }
    }

    private var alertContent: some View {
        ScrollView {
            VStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                    .font(.title2)

                Text("Glucose Dropping")
                    .font(.headline)
                    .foregroundColor(.red)

                Text(String(format: "%.0f → %.0f mg/dL",
                            store.currentGlucose,
                            store.prediction30min ?? store.currentGlucose))
                    .font(.caption)

                Text("Suggested: 15g fast carbs")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)

                // Last 4 readings for verification
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(store.recentReadings.suffix(4)) { r in
                        HStack {
                            Text(r.timestamp, style: .time)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(String(format: "%.0f", r.value))
                                .font(.caption2.bold())
                                .foregroundColor(r.value < 70 ? .red : .primary)
                        }
                    }
                }
                .padding(6)
                .background(Color.secondary.opacity(0.1))
                .cornerRadius(6)

                Button("I'm treating it") {
                    acknowledged = true
                    WKInterfaceDevice.current().play(.success)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }
            .padding(.horizontal, 8)
        }
        .onAppear { WKInterfaceDevice.current().play(.notification) }
    }

    private var safeContent: some View {
        VStack(spacing: 8) {
            Image(systemName: "checkmark.shield.fill")
                .foregroundColor(.green)
                .font(.title2)
            Text("No Alerts")
                .font(.headline)
            Text(String(format: "Current: %.0f mg/dL", store.currentGlucose))
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}
