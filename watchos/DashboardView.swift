// DashboardView.swift — Screen 1
// Live glucose reading, trend arrow, sparkline, and 30/60-min prediction tiles.
// Mirrors the GlucoSense AI Watch Screen 1 spec.

import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var store: GlucoseStore

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {

                // ── Current glucose + trend arrow ────────────────────────
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(String(format: "%.0f", store.currentGlucose))
                        .font(.system(size: 42, weight: .bold, design: .rounded))
                        .foregroundColor(glucoseColor)
                    VStack(alignment: .leading, spacing: 0) {
                        Text(trendArrow)
                            .font(.title3)
                        Text("mg/dL")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                // ── Sparkline — last 3 hours ─────────────────────────────
                if #available(watchOS 9.0, *) {
                    Chart {
                        // Safe range band
                        RectangleMark(
                            xStart: .value("Start", store.recentReadings.first?.timestamp ?? Date()),
                            xEnd: .value("End", Date()),
                            yStart: .value("Low", 70),
                            yEnd: .value("High", 180)
                        )
                        .foregroundStyle(Color.green.opacity(0.12))

                        ForEach(store.recentReadings) { reading in
                            LineMark(
                                x: .value("Time", reading.timestamp),
                                y: .value("Glucose", reading.value)
                            )
                            .foregroundStyle(Color.green)
                            .lineStyle(StrokeStyle(lineWidth: 2))
                        }
                    }
                    .chartYScale(domain: 40...350)
                    .chartXAxis(.hidden)
                    .chartYAxis(.hidden)
                    .frame(height: 50)
                }

                // ── Prediction tiles ─────────────────────────────────────
                HStack(spacing: 8) {
                    PredictionTile(
                        label: "30 min",
                        value: store.prediction30min,
                        delta: (store.prediction30min ?? 0) - store.currentGlucose
                    )
                    PredictionTile(
                        label: "60 min",
                        value: store.prediction60min,
                        delta: (store.prediction60min ?? 0) - store.currentGlucose
                    )
                }
            }
            .padding(.horizontal, 8)
        }
        .onAppear { store.fetchHistory() }
    }

    private var glucoseColor: Color {
        switch store.currentGlucose {
        case ..<54:   return .red
        case 54..<70: return .orange
        case 70...180: return .green
        case 181...250: return .yellow
        default:      return .red
        }
    }

    private var trendArrow: String {
        switch store.rateOfChange {
        case let r where r > 2:  return "↑↑"
        case let r where r > 0.5: return "↑"
        case let r where r < -2: return "↓↓"
        case let r where r < -0.5: return "↓"
        default: return "→"
        }
    }
}

struct PredictionTile: View {
    let label: String
    let value: Double?
    let delta: Double

    var body: some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            if let value {
                Text(String(format: "%.0f", value))
                    .font(.system(size: 16, weight: .semibold))
                Text(String(format: "%+.0f", delta))
                    .font(.caption2)
                    .foregroundColor(delta > 0 ? .orange : .green)
            } else {
                Text("--")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(6)
        .background(Color.secondary.opacity(0.15))
        .cornerRadius(8)
    }
}
