import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Activity {
  id: string
  name: string
  description: string
  location: string
  startTime: string
  endTime: string
  cost: number
  category: string
}

interface TravelPlan {
  id: string
  title: string
  description: string
  destination: string
  startDate: string
  endDate: string
  budget: number
  activities: Activity[]
  status: 'draft' | 'confirmed' | 'completed'
  createdAt: string
  updatedAt: string
}

interface PlanState {
  plans: TravelPlan[]
  activePlan: TravelPlan | null
  loading: boolean
  error: string | null
}

const initialState: PlanState = {
  plans: [],
  activePlan: null,
  loading: false,
  error: null,
}

const planSlice = createSlice({
  name: 'plan',
  initialState,
  reducers: {
    setPlans: (state, action: PayloadAction<TravelPlan[]>) => {
      state.plans = action.payload
    },
    addPlan: (state, action: PayloadAction<TravelPlan>) => {
      state.plans.unshift(action.payload)
    },
    updatePlan: (state, action: PayloadAction<TravelPlan>) => {
      const index = state.plans.findIndex(p => p.id === action.payload.id)
      if (index !== -1) {
        state.plans[index] = action.payload
      }
      if (state.activePlan?.id === action.payload.id) {
        state.activePlan = action.payload
      }
    },
    deletePlan: (state, action: PayloadAction<string>) => {
      state.plans = state.plans.filter(p => p.id !== action.payload)
      if (state.activePlan?.id === action.payload) {
        state.activePlan = null
      }
    },
    setActivePlan: (state, action: PayloadAction<TravelPlan | null>) => {
      state.activePlan = action.payload
    },
    addActivity: (state, action: PayloadAction<{ planId: string; activity: Activity }>) => {
      const { planId, activity } = action.payload
      const plan = state.plans.find(p => p.id === planId)
      if (plan) {
        plan.activities.push(activity)
        plan.updatedAt = new Date().toISOString()
      }
      if (state.activePlan?.id === planId) {
        state.activePlan.activities.push(activity)
        state.activePlan.updatedAt = new Date().toISOString()
      }
    },
    updateActivity: (state, action: PayloadAction<{ planId: string; activity: Activity }>) => {
      const { planId, activity } = action.payload
      const plan = state.plans.find(p => p.id === planId)
      if (plan) {
        const activityIndex = plan.activities.findIndex(a => a.id === activity.id)
        if (activityIndex !== -1) {
          plan.activities[activityIndex] = activity
          plan.updatedAt = new Date().toISOString()
        }
      }
      if (state.activePlan?.id === planId) {
        const activityIndex = state.activePlan.activities.findIndex(a => a.id === activity.id)
        if (activityIndex !== -1) {
          state.activePlan.activities[activityIndex] = activity
          state.activePlan.updatedAt = new Date().toISOString()
        }
      }
    },
    deleteActivity: (state, action: PayloadAction<{ planId: string; activityId: string }>) => {
      const { planId, activityId } = action.payload
      const plan = state.plans.find(p => p.id === planId)
      if (plan) {
        plan.activities = plan.activities.filter(a => a.id !== activityId)
        plan.updatedAt = new Date().toISOString()
      }
      if (state.activePlan?.id === planId) {
        state.activePlan.activities = state.activePlan.activities.filter(a => a.id !== activityId)
        state.activePlan.updatedAt = new Date().toISOString()
      }
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload
    },
    clearError: (state) => {
      state.error = null
    },
  },
})

export const {
  setPlans,
  addPlan,
  updatePlan,
  deletePlan,
  setActivePlan,
  addActivity,
  updateActivity,
  deleteActivity,
  setLoading,
  setError,
  clearError,
} = planSlice.actions

export default planSlice.reducer 