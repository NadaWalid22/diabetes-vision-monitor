# GlucoSense AI — watchOS Integration

## Setup steps in Xcode

1. **New project**: File → New → Project → watchOS App (standalone)
   - Product name: GlucoSense
   - Bundle ID: com.yourname.glucosense
   - Minimum deployment: watchOS 9.0

2. **Add HealthKit capability**:
   Xcode → Target → Signing & Capabilities → + HealthKit

3. **Add Core ML model**:
   - Run `python export/export_coreml.py --model lstm`
   - Drag `export/GlucoSense_lstm.mlpackage` into the Xcode project
   - Tick "Add to target: GlucoSense WatchKit Extension"

4. **Add Swift files** from `watchos/`:
   - GlucoSenseApp.swift
   - GlucoseStore.swift
   - ContentView.swift
   - DashboardView.swift
   - AlertView.swift

5. **Info.plist**: add HealthKit usage description:
   ```
   NSHealthUpdateUsageDescription  → Not required (read only)
   NSHealthShareUsageDescription   → "GlucoSense reads your CGM data to predict future glucose levels."
   ```

## Architecture

```
HealthKit (CGM readings)
        ↓
GlucoseStore (ObservableObject)
  • fetchHistory() — last 24 readings via HKSampleQuery
  • startObserving() — HKObserverQuery fires on every new CGM sample
  • runInference() — Core ML prediction → prediction30min / prediction60min
        ↓
SwiftUI Views (4 screens, TabView/page style)
  Screen 1 — DashboardView  (live reading + sparkline + prediction tiles)
  Screen 2 — TrendView      (24h chart — use Swift Charts)
  Screen 3 — AIAssistantView (conversational suggestions)
  Screen 4 — AlertView      (nocturnal hypoglycemia detection)
```

## CGM data availability

| Source           | HealthKit type                              | Notes                              |
|-----------------|---------------------------------------------|-------------------------------------|
| Dexcom G6/G7    | `HKQuantityType(.bloodGlucose)`             | Writes to HK every 5 min via iPhone |
| Libre 3         | `HKQuantityType(.bloodGlucose)`             | Direct Watch support from Libre 3+  |
| Libre 2 (EU)    | `HKQuantityType(.bloodGlucose)`             | Needs companion iPhone app          |
| Simulator       | Use HKHealthStore mock or inject test data  | See GlucoseStoreMock below          |

## Testing without a real CGM

In the Simulator, HealthKit data cannot be injected automatically.
Use `GlucoseStoreMock` (preview provider pattern):

```swift
class GlucoseStoreMock: GlucoseStore {
    override init() {
        super.init()
        currentGlucose = 131
        rateOfChange = -0.8
        prediction30min = 118
        prediction60min = 104
        recentReadings = (0..<24).map {
            GlucoseReading(
                timestamp: Date().addingTimeInterval(Double($0 - 24) * 300),
                value: 131 + sin(Double($0) / 4) * 15
            )
        }
    }
}

// In preview:
DashboardView().environmentObject(GlucoseStoreMock())
```

## Model size on device

| Model        | Parameters | Size (float32) | Apple Watch series |
|-------------|-----------|----------------|-------------------|
| LSTM         | ~50k      | ~200 KB        | S4+ (watchOS 7+)  |
| Temporal CNN | ~30k      | ~120 KB        | S4+ (watchOS 7+)  |

Both comfortably fit within watchOS memory constraints (~50 MB for extensions).
Neural Engine acceleration available on Series 7+ via `CPU_AND_NE` compute units.
