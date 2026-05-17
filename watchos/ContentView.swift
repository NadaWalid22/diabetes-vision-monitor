// ContentView.swift
// Tab navigation across the 4 GlucoSense watch screens.

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var store: GlucoseStore

    var body: some View {
        TabView {
            DashboardView()
            TrendView()
            AIAssistantView()
            AlertView()
        }
        .tabViewStyle(.page)
    }
}
