// GlucoSenseApp.swift
// Entry point for the GlucoSense AI watchOS app.
// Requires: HealthKit entitlement in Xcode project capabilities.

import SwiftUI
import HealthKit

@main
struct GlucoSenseApp: App {
    @StateObject private var glucoseStore = GlucoseStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(glucoseStore)
        }
    }
}
