# Requirement ID: FR_hybrid_1
- Description: [The system shall provide categorized access to meditation content so the user can browse meditation sessions by content type.]

- Source Persona: [Daily Calm User]
- Traceability: [Derived from review group H1]
- Acceptance Criteria:[Given the user is on the home screen, When they select a content category, Then the system displays a list of relevant content.]

# Requirement ID: FR_hybrid_2
- Description: [The system shall allow the user to filter meditation sessions by duration.]

- Source Persona: [Daily Calm User]
- Traceability: [Derived from review group H1]
- Acceptance Criteria:[Given the user is viewing the meditation library, When they apply a duration filter, Then the system shall display only meditation sessions matching the selected duration.]

# Requirement ID: FR_hybrid_3
- Description: [The system shall provide access to sleep specific content, including sleep stories, sleep meditations, or calming audio.]

- Source Persona: [Persona for Sleep Improvement with Calm]
- Traceability: [Derived from review group H2]
- Acceptance Criteria:[Given the user is on the sleep section, When they browse available sleep content, Then the system shall display sleep-related audio options.]

# Requirement ID: FR_hybrid_4
- Description: [The system shall offer a simple and minimal setup process for the user to access sleep content quickly.]

- Source Persona: [Persona for Sleep Improvement with Calm]
- Traceability: [Derived from review group H2]
- Acceptance Criteria:[Given the user is on the onboarding screen, When they select their preferred sleep content, Then the system takes them directly to the selected content.]

# Requirement ID: FR_hybrid_5
- Description: [The system shall provide the user with access to a limited but meaningful amount of free content to try before subscribing.]

- Source Persona: [Free User]
- Traceability: [Derived from review group H3]
- Acceptance Criteria:[Given the free user is browsing the content library, When content items are displayed, Then the system shall distinguish free items from subscription-locked items.]

# Requirement ID: FR_hybrid_6
- Description: [The system shall present subscription information when the user selects locked content]

- Source Persona: [Free User]
- Traceability: [Derived from review group H3]
- Acceptance Criteria:[Given the free user selects a locked content item, When access is denied, Then the system shall display that the item requires a subscription and present available subscription options.]

# Requirement ID: FR_hybrid_7
- Description: [The system shall preserve the current playback state or session progress when temporary interruptions occur during use.]

- Source Persona: [Frustrated User with Technical Issues]
- Traceability: [Derived from review group H4]
- Acceptance Criteria:[Given the user is playing audio content, When the app is temporarily interrupted and resumed, Then the system shall restore the current session at the last known playback position.]

# Requirement ID: FR_hybrid_8
- Description: [The system shall display the user’s current subscription status, billing plan, and renewal information.]

- Source Persona: [Concerened Subscriber]
- Traceability: [Derived from review group H5]
- Acceptance Criteria:[Given the user opens the subscription management screen, When the subscription details are loaded, Then the system shall display the current plan, subscription status, and next renewal or expiry date.]

# Requirement ID: FR_hybrid_9
- Description: [The system shall provide access to subscription cancellation controls through the account or subscription management interface.]

- Source Persona: [Concerened Subscriber]
- Traceability: [Derived from review group H5]
- Acceptance Criteria:[Given the user is on the subscription management screen, When they choose to manage their subscription, Then the system shall provide a visible path to cancellation instructions or cancellation controls.]

# Requirement ID: FR_hybrid_10
- Description: [The system shall inform the user of active subscription status.]

- Source Persona: [Concerened Subscriber]
- Traceability: [Derived from review group H5]
- Acceptance Criteria:[Given the user has an active trial or subscription, When they view their subscription details, Then the system shall display whether a trial is active and whether automatic renewal is scheduled.]