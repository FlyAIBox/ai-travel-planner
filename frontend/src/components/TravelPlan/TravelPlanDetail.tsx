import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Fab,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  Alert,
  Snackbar,
  CircularProgress,
  Tooltip,
  Menu,
  MenuList,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  Flight as FlightIcon,
  Hotel as HotelIcon,
  Attractions as AttractionsIcon,
  Restaurant as RestaurantIcon,
  DirectionsCar as CarIcon,
  Schedule as ScheduleIcon,
  LocationOn as LocationIcon,
  CalendarToday as CalendarIcon,
  AttachMoney as MoneyIcon,
  Share as ShareIcon,
  Print as PrintIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  Visibility as ViewIcon,
  VisibilityOff as HideIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  Weather as WeatherIcon,
  Navigation as NavigationIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { DatePicker, TimePicker } from '@mui/x-date-pickers';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// 类型定义
interface Location {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  city: string;
  country: string;
  category: string;
  rating: number;
  cost_level: number;
  visit_duration: number;
  opening_hours: Record<string, string>;
}

interface Activity {
  id: string;
  name: string;
  location: Location;
  category: string;
  duration: number;
  cost: number;
  rating: number;
  description: string;
  requirements: string[];
  best_time: string[];
  start_time?: string;
  end_time?: string;
  status: 'pending' | 'confirmed' | 'completed' | 'cancelled';
  weather_dependent?: boolean;
  booking_required?: boolean;
  notes?: string;
}

interface Transportation {
  id: string;
  mode: string;
  from_location: Location;
  to_location: Location;
  duration: number;
  cost: number;
  departure_time?: string;
  arrival_time?: string;
  booking_required: boolean;
  booking_status?: 'pending' | 'confirmed' | 'cancelled';
  provider?: string;
  booking_reference?: string;
}

interface DayPlan {
  date: string;
  activities: Activity[];
  transportations: Transportation[];
  start_time: string;
  end_time: string;
  budget_used: number;
  budget_limit: number;
  notes?: string;
  weather_forecast?: any;
}

interface TravelPlan {
  id: string;
  name: string;
  description: string;
  start_date: string;
  end_date: string;
  daily_plans: DayPlan[];
  total_cost: number;
  status: 'draft' | 'ready' | 'confirmed' | 'in_progress' | 'completed' | 'cancelled';
  participants: string[];
  preferences: Record<string, any>;
  created_at: string;
  updated_at: string;
}

const TravelPlanDetail: React.FC = () => {
  const { planId } = useParams<{ planId: string }>();
  const navigate = useNavigate();
  
  // State管理
  const [plan, setPlan] = useState<TravelPlan | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [selectedDay, setSelectedDay] = useState(0);
  const [viewMode, setViewMode] = useState<'timeline' | 'cards' | 'map'>('timeline');
  
  // 对话框状态
  const [activityDialogOpen, setActivityDialogOpen] = useState(false);
  const [transportDialogOpen, setTransportDialogOpen] = useState(false);
  const [editingActivity, setEditingActivity] = useState<Activity | null>(null);
  const [editingTransport, setEditingTransport] = useState<Transportation | null>(null);
  
  // 菜单状态
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  
  // 通知状态
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'warning' | 'info'
  });

  // 加载计划数据
  const loadPlan = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`/api/v1/planning/${planId}`);
      if (!response.ok) {
        throw new Error('Failed to load travel plan');
      }
      
      const planData = await response.json();
      setPlan(planData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error loading plan:', err);
    } finally {
      setLoading(false);
    }
  }, [planId]);

  // 保存计划
  const savePlan = useCallback(async (updatedPlan: TravelPlan) => {
    try {
      setSaving(true);
      
      const response = await fetch(`/api/v1/planning/${planId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedPlan),
      });

      if (!response.ok) {
        throw new Error('Failed to save travel plan');
      }

      const savedPlan = await response.json();
      setPlan(savedPlan);
      
      showSnackbar('计划已保存', 'success');
    } catch (err) {
      showSnackbar('保存失败: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error');
    } finally {
      setSaving(false);
    }
  }, [planId]);

  // 显示通知
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  // 添加活动
  const addActivity = useCallback(async (dayIndex: number, activityData: Partial<Activity>) => {
    if (!plan) return;

    const newActivity: Activity = {
      id: `activity_${Date.now()}`,
      name: activityData.name || '',
      location: activityData.location!,
      category: activityData.category || 'general',
      duration: activityData.duration || 120,
      cost: activityData.cost || 0,
      rating: activityData.rating || 0,
      description: activityData.description || '',
      requirements: activityData.requirements || [],
      best_time: activityData.best_time || [],
      status: 'pending',
      ...activityData,
    };

    const updatedPlan = { ...plan };
    updatedPlan.daily_plans[dayIndex].activities.push(newActivity);
    
    await savePlan(updatedPlan);
    setActivityDialogOpen(false);
    setEditingActivity(null);
  }, [plan, savePlan]);

  // 更新活动
  const updateActivity = useCallback(async (dayIndex: number, activityIndex: number, activityData: Partial<Activity>) => {
    if (!plan) return;

    const updatedPlan = { ...plan };
    updatedPlan.daily_plans[dayIndex].activities[activityIndex] = {
      ...updatedPlan.daily_plans[dayIndex].activities[activityIndex],
      ...activityData,
    };
    
    await savePlan(updatedPlan);
    setActivityDialogOpen(false);
    setEditingActivity(null);
  }, [plan, savePlan]);

  // 删除活动
  const deleteActivity = useCallback(async (dayIndex: number, activityIndex: number) => {
    if (!plan) return;

    const updatedPlan = { ...plan };
    updatedPlan.daily_plans[dayIndex].activities.splice(activityIndex, 1);
    
    await savePlan(updatedPlan);
    showSnackbar('活动已删除', 'success');
  }, [plan, savePlan]);

  // 优化行程
  const optimizePlan = useCallback(async () => {
    try {
      setSaving(true);
      
      const response = await fetch(`/api/v1/planning/${planId}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to optimize plan');
      }

      const optimizedPlan = await response.json();
      setPlan(optimizedPlan);
      
      showSnackbar('行程已优化', 'success');
    } catch (err) {
      showSnackbar('优化失败: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error');
    } finally {
      setSaving(false);
    }
  }, [planId]);

  useEffect(() => {
    if (planId) {
      loadPlan();
    }
  }, [planId, loadPlan]);

  // 渲染活动卡片
  const renderActivityCard = (activity: Activity, dayIndex: number, activityIndex: number) => {
    const categoryIcons = {
      flight: <FlightIcon />,
      hotel: <HotelIcon />,
      attraction: <AttractionsIcon />,
      restaurant: <RestaurantIcon />,
      transport: <CarIcon />,
    };

    const statusColors = {
      pending: 'default',
      confirmed: 'primary',
      completed: 'success',
      cancelled: 'error',
    } as const;

    return (
      <motion.div
        key={activity.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >
        <Card 
          sx={{ 
            mb: 2, 
            border: activity.status === 'completed' ? '2px solid #4caf50' : 'none',
            opacity: activity.status === 'cancelled' ? 0.6 : 1,
          }}
        >
          <CardContent>
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
              <Box display="flex" alignItems="center" gap={1}>
                {categoryIcons[activity.category as keyof typeof categoryIcons] || <InfoIcon />}
                <Typography variant="h6" component="h3">
                  {activity.name}
                </Typography>
                <Chip 
                  label={activity.status} 
                  color={statusColors[activity.status]} 
                  size="small" 
                />
              </Box>
              
              {editMode && (
                <Box>
                  <IconButton
                    size="small"
                    onClick={() => {
                      setEditingActivity(activity);
                      setActivityDialogOpen(true);
                    }}
                  >
                    <EditIcon />
                  </IconButton>
                  <IconButton
                    size="small"
                    onClick={() => deleteActivity(dayIndex, activityIndex)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              )}
            </Box>

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <LocationIcon fontSize="small" />
                  <Typography variant="body2" color="text.secondary">
                    {activity.location.name}
                  </Typography>
                </Box>
                
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <ScheduleIcon fontSize="small" />
                  <Typography variant="body2" color="text.secondary">
                    {activity.duration} 分钟
                  </Typography>
                  {activity.start_time && (
                    <Typography variant="body2" color="text.secondary">
                      ({activity.start_time} - {activity.end_time})
                    </Typography>
                  )}
                </Box>

                <Box display="flex" alignItems="center" gap={1}>
                  <MoneyIcon fontSize="small" />
                  <Typography variant="body2" color="text.secondary">
                    ¥{activity.cost}
                  </Typography>
                  {activity.rating > 0 && (
                    <Box display="flex" alignItems="center" ml={2}>
                      <StarIcon fontSize="small" color="warning" />
                      <Typography variant="body2" color="text.secondary">
                        {activity.rating}
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                {activity.description && (
                  <Typography variant="body2" color="text.secondary" mb={1}>
                    {activity.description}
                  </Typography>
                )}
                
                {activity.requirements.length > 0 && (
                  <Box mb={1}>
                    <Typography variant="caption" color="text.secondary">
                      要求: {activity.requirements.join(', ')}
                    </Typography>
                  </Box>
                )}

                {activity.notes && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      备注: {activity.notes}
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>

          {activity.booking_required && (
            <CardActions>
              <Button size="small" variant="outlined">
                立即预订
              </Button>
              <Button size="small">
                查看详情
              </Button>
            </CardActions>
          )}
        </Card>
      </motion.div>
    );
  };

  // 渲染日程时间线
  const renderTimelineView = (dayPlan: DayPlan, dayIndex: number) => {
    const allItems = [
      ...dayPlan.activities.map(activity => ({ type: 'activity', data: activity })),
      ...dayPlan.transportations.map(transport => ({ type: 'transport', data: transport })),
    ].sort((a, b) => {
      const timeA = a.type === 'activity' ? a.data.start_time : a.data.departure_time;
      const timeB = b.type === 'activity' ? b.data.start_time : b.data.departure_time;
      return (timeA || '').localeCompare(timeB || '');
    });

    return (
      <Timeline>
        {allItems.map((item, index) => (
          <TimelineItem key={index}>
            <TimelineSeparator>
              <TimelineDot 
                color={item.type === 'activity' ? 'primary' : 'secondary'}
              >
                {item.type === 'activity' ? <AttractionsIcon /> : <CarIcon />}
              </TimelineDot>
              {index < allItems.length - 1 && <TimelineConnector />}
            </TimelineSeparator>
            
            <TimelineContent>
              {item.type === 'activity' ? 
                renderActivityCard(item.data as Activity, dayIndex, 
                  dayPlan.activities.findIndex(a => a.id === item.data.id)) :
                renderTransportCard(item.data as Transportation)
              }
            </TimelineContent>
          </TimelineItem>
        ))}
      </Timeline>
    );
  };

  // 渲染交通卡片
  const renderTransportCard = (transport: Transportation) => {
    return (
      <Card sx={{ mb: 1 }}>
        <CardContent sx={{ py: 1 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={1}>
              <CarIcon fontSize="small" />
              <Typography variant="body2">
                {transport.from_location.name} → {transport.to_location.name}
              </Typography>
            </Box>
            
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="caption" color="text.secondary">
                {transport.duration} 分钟
              </Typography>
              <Typography variant="caption" color="text.secondary">
                ¥{transport.cost}
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  };

  // 渲染日计划概览
  const renderDayOverview = (dayPlan: DayPlan, dayIndex: number) => {
    const totalCost = dayPlan.activities.reduce((sum, activity) => sum + activity.cost, 0) +
                    dayPlan.transportations.reduce((sum, transport) => sum + transport.cost, 0);
    const totalDuration = dayPlan.activities.reduce((sum, activity) => sum + activity.duration, 0);
    const budgetUsage = dayPlan.budget_limit > 0 ? (totalCost / dayPlan.budget_limit) * 100 : 0;

    return (
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <Typography variant="h6">
              第 {dayIndex + 1} 天
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {new Date(dayPlan.date).toLocaleDateString('zh-CN', {
                month: 'long',
                day: 'numeric',
                weekday: 'short'
              })}
            </Typography>
          </Grid>

          <Grid item xs={12} md={3}>
            <Typography variant="body2" color="text.secondary">
              活动数量
            </Typography>
            <Typography variant="h6">
              {dayPlan.activities.length} 个
            </Typography>
          </Grid>

          <Grid item xs={12} md={3}>
            <Typography variant="body2" color="text.secondary">
              总费用
            </Typography>
            <Typography variant="h6" color={budgetUsage > 100 ? 'error' : 'text.primary'}>
              ¥{totalCost}
            </Typography>
            {dayPlan.budget_limit > 0 && (
              <Typography variant="caption" color="text.secondary">
                / ¥{dayPlan.budget_limit} ({budgetUsage.toFixed(1)}%)
              </Typography>
            )}
          </Grid>

          <Grid item xs={12} md={3}>
            <Typography variant="body2" color="text.secondary">
              总时长
            </Typography>
            <Typography variant="h6">
              {Math.floor(totalDuration / 60)}h {totalDuration % 60}m
            </Typography>
          </Grid>
        </Grid>

        {dayPlan.weather_forecast && (
          <Box mt={2} display="flex" alignItems="center" gap={1}>
            <WeatherIcon fontSize="small" />
            <Typography variant="body2" color="text.secondary">
              {dayPlan.weather_forecast.condition}, {dayPlan.weather_forecast.temperature}°C
            </Typography>
          </Box>
        )}
      </Paper>
    );
  };

  // 渲染活动编辑对话框
  const renderActivityDialog = () => {
    return (
      <Dialog 
        open={activityDialogOpen} 
        onClose={() => setActivityDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingActivity ? '编辑活动' : '添加活动'}
        </DialogTitle>
        <DialogContent>
          {/* Activity form fields */}
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="活动名称"
                defaultValue={editingActivity?.name || ''}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>活动类型</InputLabel>
                <Select defaultValue={editingActivity?.category || 'attraction'}>
                  <MenuItem value="attraction">景点</MenuItem>
                  <MenuItem value="restaurant">餐厅</MenuItem>
                  <MenuItem value="hotel">酒店</MenuItem>
                  <MenuItem value="transport">交通</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="预计费用"
                type="number"
                defaultValue={editingActivity?.cost || 0}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="活动描述"
                defaultValue={editingActivity?.description || ''}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setActivityDialogOpen(false)}>
            取消
          </Button>
          <Button 
            variant="contained" 
            onClick={() => {
              // Handle save logic
              setActivityDialogOpen(false);
            }}
          >
            保存
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">
          {error}
          <Button onClick={loadPlan} sx={{ ml: 2 }}>
            重试
          </Button>
        </Alert>
      </Container>
    );
  }

  if (!plan) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="warning">
          未找到旅行计划
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              {plan.name}
            </Typography>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              {plan.description}
            </Typography>
            <Box display="flex" alignItems="center" gap={2} mt={1}>
              <Chip 
                label={plan.status} 
                color={plan.status === 'ready' ? 'primary' : 'default'} 
              />
              <Typography variant="body2" color="text.secondary">
                {new Date(plan.start_date).toLocaleDateString()} - {new Date(plan.end_date).toLocaleDateString()}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                总费用: ¥{plan.total_cost}
              </Typography>
            </Box>
          </Box>

          <Box display="flex" gap={1}>
            <Tooltip title="查看模式">
              <Button
                variant={viewMode === 'timeline' ? 'contained' : 'outlined'}
                onClick={() => setViewMode('timeline')}
              >
                时间线
              </Button>
            </Tooltip>
            <Tooltip title="卡片模式">
              <Button
                variant={viewMode === 'cards' ? 'contained' : 'outlined'}
                onClick={() => setViewMode('cards')}
              >
                卡片
              </Button>
            </Tooltip>
            
            <Button
              variant={editMode ? 'contained' : 'outlined'}
              onClick={() => setEditMode(!editMode)}
              startIcon={<EditIcon />}
            >
              {editMode ? '完成编辑' : '编辑计划'}
            </Button>

            <IconButton
              onClick={(e) => {
                setMenuAnchorEl(e.currentTarget);
                setActionMenuOpen(true);
              }}
            >
              <SettingsIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Action Menu */}
        <Menu
          anchorEl={menuAnchorEl}
          open={actionMenuOpen}
          onClose={() => setActionMenuOpen(false)}
        >
          <MenuList>
            <MenuItem onClick={optimizePlan} disabled={saving}>
              <ListItemIcon>
                <NavigationIcon />
              </ListItemIcon>
              <ListItemText>优化行程</ListItemText>
            </MenuItem>
            <MenuItem>
              <ListItemIcon>
                <ShareIcon />
              </ListItemIcon>
              <ListItemText>分享计划</ListItemText>
            </MenuItem>
            <MenuItem>
              <ListItemIcon>
                <PrintIcon />
              </ListItemIcon>
              <ListItemText>打印行程</ListItemText>
            </MenuItem>
            <MenuItem>
              <ListItemIcon>
                <DownloadIcon />
              </ListItemIcon>
              <ListItemText>导出PDF</ListItemText>
            </MenuItem>
          </MenuList>
        </Menu>
      </Paper>

      {/* Day Navigation */}
      <Paper sx={{ mb: 3 }}>
        <Stepper activeStep={selectedDay} orientation="horizontal" sx={{ p: 2 }}>
          {plan.daily_plans.map((dayPlan, index) => (
            <Step key={index} onClick={() => setSelectedDay(index)} sx={{ cursor: 'pointer' }}>
              <StepLabel>
                第 {index + 1} 天
                <Typography variant="caption" display="block">
                  {new Date(dayPlan.date).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })}
                </Typography>
              </StepLabel>
            </Step>
          ))}
        </Stepper>
      </Paper>

      {/* Day Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedDay}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          {plan.daily_plans[selectedDay] && (
            <>
              {renderDayOverview(plan.daily_plans[selectedDay], selectedDay)}
              
              {viewMode === 'timeline' ? 
                renderTimelineView(plan.daily_plans[selectedDay], selectedDay) :
                <Box>
                  {plan.daily_plans[selectedDay].activities.map((activity, activityIndex) =>
                    renderActivityCard(activity, selectedDay, activityIndex)
                  )}
                </Box>
              }
            </>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Floating Action Button */}
      {editMode && (
        <Fab
          color="primary"
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={() => setActivityDialogOpen(true)}
        >
          <AddIcon />
        </Fab>
      )}

      {/* Dialogs */}
      {renderActivityDialog()}

      {/* Snackbar */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          severity={snackbar.severity}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      {/* Loading Overlay */}
      {saving && (
        <Box
          position="fixed"
          top={0}
          left={0}
          right={0}
          bottom={0}
          bgcolor="rgba(0,0,0,0.3)"
          display="flex"
          alignItems="center"
          justifyContent="center"
          zIndex={9999}
        >
          <CircularProgress />
        </Box>
      )}
    </Container>
  );
};

export default TravelPlanDetail; 