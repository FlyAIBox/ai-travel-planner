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
  Fab,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Menu,
  MenuList,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Alert,
  Snackbar,
  CircularProgress,
  Tooltip,
  InputAdornment,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  ButtonGroup,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
} from '@mui/material';
import {
  Add as AddIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Sort as SortIcon,
  MoreVert as MoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Share as ShareIcon,
  FileCopy as CopyIcon,
  Visibility as ViewIcon,
  CalendarToday as CalendarIcon,
  AttachMoney as MoneyIcon,
  LocationOn as LocationIcon,
  People as PeopleIcon,
  Flight as FlightIcon,
  Hotel as HotelIcon,
  Schedule as ScheduleIcon,
  Star as StarIcon,
  TrendingUp as TrendingUpIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

// 类型定义
interface TravelPlanSummary {
  id: string;
  name: string;
  description: string;
  start_date: string;
  end_date: string;
  status: 'draft' | 'ready' | 'confirmed' | 'in_progress' | 'completed' | 'cancelled';
  total_cost: number;
  duration_days: number;
  destinations: string[];
  participants: number;
  created_at: string;
  updated_at: string;
  thumbnail?: string;
  tags?: string[];
  is_favorite?: boolean;
  shared_with?: string[];
}

interface FilterCriteria {
  status: string[];
  dateRange: {
    start: Date | null;
    end: Date | null;
  };
  budgetRange: {
    min: number;
    max: number;
  };
  destinations: string[];
  duration: {
    min: number;
    max: number;
  };
}

interface SortOption {
  field: 'created_at' | 'start_date' | 'total_cost' | 'duration_days' | 'name';
  direction: 'asc' | 'desc';
}

const TravelPlanList: React.FC = () => {
  const navigate = useNavigate();
  
  // State管理
  const [plans, setPlans] = useState<TravelPlanSummary[]>([]);
  const [filteredPlans, setFilteredPlans] = useState<TravelPlanSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // 搜索和筛选
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCriteria, setFilterCriteria] = useState<FilterCriteria>({
    status: [],
    dateRange: { start: null, end: null },
    budgetRange: { min: 0, max: 100000 },
    destinations: [],
    duration: { min: 1, max: 30 },
  });
  const [sortOption, setSortOption] = useState<SortOption>({
    field: 'created_at',
    direction: 'desc',
  });
  
  // UI状态
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [selectedPlans, setSelectedPlans] = useState<string[]>([]);
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [menuAnchorEl, setMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(null);
  
  // 通知状态
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success' as 'success' | 'error' | 'warning' | 'info'
  });

  // 加载计划列表
  const loadPlans = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/v1/planning/plans');
      if (!response.ok) {
        throw new Error('Failed to load travel plans');
      }
      
      const plansData = await response.json();
      setPlans(plansData);
      setFilteredPlans(plansData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error loading plans:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // 应用筛选和排序
  const applyFiltersAndSort = useCallback(() => {
    let filtered = [...plans];
    
    // 搜索过滤
    if (searchTerm) {
      filtered = filtered.filter(plan =>
        plan.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        plan.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        plan.destinations.some(dest => dest.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }
    
    // 状态过滤
    if (filterCriteria.status.length > 0) {
      filtered = filtered.filter(plan => filterCriteria.status.includes(plan.status));
    }
    
    // 日期范围过滤
    if (filterCriteria.dateRange.start && filterCriteria.dateRange.end) {
      filtered = filtered.filter(plan => {
        const planStart = new Date(plan.start_date);
        return planStart >= filterCriteria.dateRange.start! && 
               planStart <= filterCriteria.dateRange.end!;
      });
    }
    
    // 预算范围过滤
    filtered = filtered.filter(plan => 
      plan.total_cost >= filterCriteria.budgetRange.min && 
      plan.total_cost <= filterCriteria.budgetRange.max
    );
    
    // 时长过滤
    filtered = filtered.filter(plan => 
      plan.duration_days >= filterCriteria.duration.min && 
      plan.duration_days <= filterCriteria.duration.max
    );
    
    // 排序
    filtered.sort((a, b) => {
      const aVal = a[sortOption.field];
      const bVal = b[sortOption.field];
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortOption.direction === 'asc' ? 
          aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      } else if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortOption.direction === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      return 0;
    });
    
    setFilteredPlans(filtered);
  }, [plans, searchTerm, filterCriteria, sortOption]);

  // 创建新计划
  const createNewPlan = useCallback(async (planData: any) => {
    try {
      const response = await fetch('/api/v1/planning/plans', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(planData),
      });

      if (!response.ok) {
        throw new Error('Failed to create travel plan');
      }

      const newPlan = await response.json();
      
      showSnackbar('计划创建成功', 'success');
      setCreateDialogOpen(false);
      loadPlans(); // 重新加载列表
      
      // 导航到新计划详情页
      navigate(`/plans/${newPlan.id}`);
    } catch (err) {
      showSnackbar('创建失败: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error');
    }
  }, [navigate]);

  // 删除计划
  const deletePlan = useCallback(async (planId: string) => {
    try {
      const response = await fetch(`/api/v1/planning/plans/${planId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete travel plan');
      }

      showSnackbar('计划已删除', 'success');
      loadPlans(); // 重新加载列表
    } catch (err) {
      showSnackbar('删除失败: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error');
    }
  }, []);

  // 复制计划
  const duplicatePlan = useCallback(async (planId: string) => {
    try {
      const response = await fetch(`/api/v1/planning/plans/${planId}/duplicate`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to duplicate travel plan');
      }

      const duplicatedPlan = await response.json();
      showSnackbar('计划已复制', 'success');
      loadPlans(); // 重新加载列表
      
      // 导航到复制的计划
      navigate(`/plans/${duplicatedPlan.id}`);
    } catch (err) {
      showSnackbar('复制失败: ' + (err instanceof Error ? err.message : 'Unknown error'), 'error');
    }
  }, [navigate]);

  // 显示通知
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  // 切换计划选择
  const togglePlanSelection = (planId: string) => {
    setSelectedPlans(prev => 
      prev.includes(planId) 
        ? prev.filter(id => id !== planId)
        : [...prev, planId]
    );
  };

  // 批量操作
  const handleBatchOperation = async (operation: 'delete' | 'export') => {
    if (selectedPlans.length === 0) return;
    
    if (operation === 'delete') {
      // 确认删除
      if (window.confirm(`确定要删除 ${selectedPlans.length} 个计划吗？`)) {
        for (const planId of selectedPlans) {
          await deletePlan(planId);
        }
        setSelectedPlans([]);
      }
    } else if (operation === 'export') {
      // 导出逻辑
      showSnackbar('导出功能开发中', 'info');
    }
  };

  useEffect(() => {
    loadPlans();
  }, [loadPlans]);

  useEffect(() => {
    applyFiltersAndSort();
  }, [applyFiltersAndSort]);

  // 渲染计划卡片
  const renderPlanCard = (plan: TravelPlanSummary) => {
    const statusColors = {
      draft: 'default',
      ready: 'primary',
      confirmed: 'info',
      in_progress: 'warning',
      completed: 'success',
      cancelled: 'error',
    } as const;

    const statusLabels = {
      draft: '草稿',
      ready: '就绪',
      confirmed: '已确认',
      in_progress: '进行中',
      completed: '已完成',
      cancelled: '已取消',
    };

    return (
      <motion.div
        key={plan.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >
        <Card 
          sx={{ 
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
            '&:hover': { elevation: 4 },
            cursor: 'pointer',
          }}
          onClick={() => navigate(`/plans/${plan.id}`)}
        >
          {/* 选择框 */}
          <Checkbox
            checked={selectedPlans.includes(plan.id)}
            onChange={(e) => {
              e.stopPropagation();
              togglePlanSelection(plan.id);
            }}
            sx={{ position: 'absolute', top: 8, left: 8, zIndex: 1 }}
          />

          {/* 更多操作菜单 */}
          <IconButton
            sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}
            onClick={(e) => {
              e.stopPropagation();
              setMenuAnchorEl(e.currentTarget);
              setSelectedPlanId(plan.id);
            }}
          >
            <MoreIcon />
          </IconButton>

          {/* 缩略图 */}
          {plan.thumbnail ? (
            <Box
              component="img"
              src={plan.thumbnail}
              alt={plan.name}
              sx={{ height: 140, objectFit: 'cover' }}
            />
          ) : (
            <Box
              sx={{
                height: 140,
                bgcolor: 'grey.200',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <LocationIcon sx={{ fontSize: 40, color: 'grey.400' }} />
            </Box>
          )}

          <CardContent sx={{ flexGrow: 1, pb: 1 }}>
            {/* 标题和状态 */}
            <Box display="flex" justifyContent="space-between" alignItems="start" mb={1}>
              <Typography variant="h6" component="h3" noWrap sx={{ flex: 1, mr: 1 }}>
                {plan.name}
              </Typography>
              <Chip 
                label={statusLabels[plan.status]} 
                color={statusColors[plan.status]} 
                size="small"
              />
            </Box>

            {/* 描述 */}
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ 
                mb: 2,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
              }}
            >
              {plan.description}
            </Typography>

            {/* 目的地 */}
            <Box display="flex" alignItems="center" mb={1}>
              <LocationIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary" noWrap>
                {plan.destinations.join(', ')}
              </Typography>
            </Box>

            {/* 日期 */}
            <Box display="flex" alignItems="center" mb={1}>
              <CalendarIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
              <Typography variant="body2" color="text.secondary">
                {new Date(plan.start_date).toLocaleDateString()} - {new Date(plan.end_date).toLocaleDateString()}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                ({plan.duration_days} 天)
              </Typography>
            </Box>

            {/* 费用和参与人数 */}
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Box display="flex" alignItems="center">
                <MoneyIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="body2" color="text.secondary">
                  ¥{plan.total_cost.toLocaleString()}
                </Typography>
              </Box>
              <Box display="flex" alignItems="center">
                <PeopleIcon fontSize="small" sx={{ mr: 1, color: 'text.secondary' }} />
                <Typography variant="body2" color="text.secondary">
                  {plan.participants} 人
                </Typography>
              </Box>
            </Box>

            {/* 标签 */}
            {plan.tags && plan.tags.length > 0 && (
              <Box display="flex" gap={0.5} flexWrap="wrap" mt={1}>
                {plan.tags.slice(0, 3).map((tag) => (
                  <Chip key={tag} label={tag} size="small" variant="outlined" />
                ))}
                {plan.tags.length > 3 && (
                  <Chip label={`+${plan.tags.length - 3}`} size="small" variant="outlined" />
                )}
              </Box>
            )}
          </CardContent>

          <CardActions sx={{ pt: 0 }}>
            <Button size="small" onClick={(e) => {
              e.stopPropagation();
              navigate(`/plans/${plan.id}`);
            }}>
              查看详情
            </Button>
            <Button size="small" onClick={(e) => {
              e.stopPropagation();
              navigate(`/plans/${plan.id}/edit`);
            }}>
              编辑
            </Button>
            {plan.is_favorite && (
              <StarIcon fontSize="small" color="warning" sx={{ ml: 'auto' }} />
            )}
          </CardActions>
        </Card>
      </motion.div>
    );
  };

  // 渲染筛选对话框
  const renderFilterDialog = () => {
    return (
      <Dialog open={filterDialogOpen} onClose={() => setFilterDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          筛选计划
          <IconButton
            sx={{ position: 'absolute', right: 8, top: 8 }}
            onClick={() => setFilterDialogOpen(false)}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            {/* 状态筛选 */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>状态</InputLabel>
                <Select
                  multiple
                  value={filterCriteria.status}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    status: typeof e.target.value === 'string' ? [e.target.value] : e.target.value
                  }))}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  <MenuItem value="draft">草稿</MenuItem>
                  <MenuItem value="ready">就绪</MenuItem>
                  <MenuItem value="confirmed">已确认</MenuItem>
                  <MenuItem value="in_progress">进行中</MenuItem>
                  <MenuItem value="completed">已完成</MenuItem>
                  <MenuItem value="cancelled">已取消</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            {/* 预算范围 */}
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>预算范围</Typography>
              <Box display="flex" gap={2}>
                <TextField
                  label="最低"
                  type="number"
                  value={filterCriteria.budgetRange.min}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    budgetRange: { ...prev.budgetRange, min: Number(e.target.value) }
                  }))}
                />
                <TextField
                  label="最高"
                  type="number"
                  value={filterCriteria.budgetRange.max}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    budgetRange: { ...prev.budgetRange, max: Number(e.target.value) }
                  }))}
                />
              </Box>
            </Grid>

            {/* 出行日期范围 */}
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>出行日期</Typography>
              <Box display="flex" gap={2}>
                <DatePicker
                  label="开始日期"
                  value={filterCriteria.dateRange.start}
                  onChange={(date) => setFilterCriteria(prev => ({
                    ...prev,
                    dateRange: { ...prev.dateRange, start: date }
                  }))}
                />
                <DatePicker
                  label="结束日期"
                  value={filterCriteria.dateRange.end}
                  onChange={(date) => setFilterCriteria(prev => ({
                    ...prev,
                    dateRange: { ...prev.dateRange, end: date }
                  }))}
                />
              </Box>
            </Grid>

            {/* 行程天数 */}
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>行程天数</Typography>
              <Box display="flex" gap={2}>
                <TextField
                  label="最少天数"
                  type="number"
                  value={filterCriteria.duration.min}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    duration: { ...prev.duration, min: Number(e.target.value) }
                  }))}
                />
                <TextField
                  label="最多天数"
                  type="number"
                  value={filterCriteria.duration.max}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    duration: { ...prev.duration, max: Number(e.target.value) }
                  }))}
                />
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setFilterCriteria({
              status: [],
              dateRange: { start: null, end: null },
              budgetRange: { min: 0, max: 100000 },
              destinations: [],
              duration: { min: 1, max: 30 },
            });
          }}>
            重置
          </Button>
          <Button onClick={() => setFilterDialogOpen(false)}>
            取消
          </Button>
          <Button variant="contained" onClick={() => setFilterDialogOpen(false)}>
            应用筛选
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  // 渲染创建计划对话框
  const renderCreateDialog = () => {
    const [newPlanData, setNewPlanData] = useState({
      name: '',
      description: '',
      destinations: [],
      start_date: '',
      end_date: '',
      budget: 0,
    });

    return (
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>创建新的旅行计划</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="计划名称"
                value={newPlanData.name}
                onChange={(e) => setNewPlanData(prev => ({ ...prev, name: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={3}
                label="计划描述"
                value={newPlanData.description}
                onChange={(e) => setNewPlanData(prev => ({ ...prev, description: e.target.value }))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <DatePicker
                label="开始日期"
                value={newPlanData.start_date ? new Date(newPlanData.start_date) : null}
                onChange={(date) => setNewPlanData(prev => ({ 
                  ...prev, 
                  start_date: date ? date.toISOString().split('T')[0] : '' 
                }))}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <DatePicker
                label="结束日期"
                value={newPlanData.end_date ? new Date(newPlanData.end_date) : null}
                onChange={(date) => setNewPlanData(prev => ({ 
                  ...prev, 
                  end_date: date ? date.toISOString().split('T')[0] : '' 
                }))}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="预算"
                type="number"
                value={newPlanData.budget}
                onChange={(e) => setNewPlanData(prev => ({ ...prev, budget: Number(e.target.value) }))}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>取消</Button>
          <Button 
            variant="contained" 
            onClick={() => createNewPlan(newPlanData)}
            disabled={!newPlanData.name || !newPlanData.start_date}
          >
            创建计划
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

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          我的旅行计划
        </Typography>
        
        <Box display="flex" gap={2}>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
          >
            创建计划
          </Button>
          <IconButton onClick={loadPlans}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>

      {/* 工具栏 */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          {/* 搜索 */}
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              placeholder="搜索计划..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>

          {/* 排序 */}
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>排序</InputLabel>
              <Select
                value={`${sortOption.field}-${sortOption.direction}`}
                onChange={(e) => {
                  const [field, direction] = e.target.value.split('-');
                  setSortOption({ 
                    field: field as SortOption['field'], 
                    direction: direction as SortOption['direction'] 
                  });
                }}
              >
                <MenuItem value="created_at-desc">创建时间（最新）</MenuItem>
                <MenuItem value="created_at-asc">创建时间（最旧）</MenuItem>
                <MenuItem value="start_date-desc">出行日期（最近）</MenuItem>
                <MenuItem value="start_date-asc">出行日期（最远）</MenuItem>
                <MenuItem value="total_cost-desc">费用（高到低）</MenuItem>
                <MenuItem value="total_cost-asc">费用（低到高）</MenuItem>
                <MenuItem value="name-asc">名称（A-Z）</MenuItem>
                <MenuItem value="name-desc">名称（Z-A）</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* 筛选和视图切换 */}
          <Grid item xs={12} md={5}>
            <Box display="flex" justifyContent="flex-end" gap={1}>
              <Button
                startIcon={<FilterIcon />}
                onClick={() => setFilterDialogOpen(true)}
              >
                筛选
              </Button>
              
              <ButtonGroup>
                <Button
                  variant={viewMode === 'grid' ? 'contained' : 'outlined'}
                  onClick={() => setViewMode('grid')}
                >
                  网格
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'contained' : 'outlined'}
                  onClick={() => setViewMode('list')}
                >
                  列表
                </Button>
              </ButtonGroup>
            </Box>
          </Grid>
        </Grid>

        {/* 批量操作 */}
        {selectedPlans.length > 0 && (
          <Box mt={2} display="flex" alignItems="center" gap={2}>
            <Typography variant="body2">
              已选择 {selectedPlans.length} 个计划
            </Typography>
            <Button
              size="small"
              startIcon={<DeleteIcon />}
              onClick={() => handleBatchOperation('delete')}
            >
              删除
            </Button>
            <Button
              size="small"
              startIcon={<DownloadIcon />}
              onClick={() => handleBatchOperation('export')}
            >
              导出
            </Button>
            <Button
              size="small"
              onClick={() => setSelectedPlans([])}
            >
              取消选择
            </Button>
          </Box>
        )}
      </Paper>

      {/* 错误提示 */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
          <Button onClick={loadPlans} sx={{ ml: 2 }}>
            重试
          </Button>
        </Alert>
      )}

      {/* 计划列表 */}
      {filteredPlans.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            {plans.length === 0 ? '暂无旅行计划' : '没有符合筛选条件的计划'}
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {plans.length === 0 ? '开始创建您的第一个旅行计划吧！' : '尝试调整筛选条件'}
          </Typography>
          {plans.length === 0 && (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
              sx={{ mt: 2 }}
            >
              创建计划
            </Button>
          )}
        </Paper>
      ) : (
        <Grid container spacing={3}>
          <AnimatePresence>
            {filteredPlans.map((plan) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={plan.id}>
                {renderPlanCard(plan)}
              </Grid>
            ))}
          </AnimatePresence>
        </Grid>
      )}

      {/* 更多操作菜单 */}
      <Menu
        anchorEl={menuAnchorEl}
        open={Boolean(menuAnchorEl)}
        onClose={() => setMenuAnchorEl(null)}
      >
        <MenuList>
          <MenuItem onClick={() => {
            if (selectedPlanId) navigate(`/plans/${selectedPlanId}`);
            setMenuAnchorEl(null);
          }}>
            <ListItemIcon><ViewIcon /></ListItemIcon>
            <ListItemText>查看详情</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => {
            if (selectedPlanId) navigate(`/plans/${selectedPlanId}/edit`);
            setMenuAnchorEl(null);
          }}>
            <ListItemIcon><EditIcon /></ListItemIcon>
            <ListItemText>编辑计划</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => {
            if (selectedPlanId) duplicatePlan(selectedPlanId);
            setMenuAnchorEl(null);
          }}>
            <ListItemIcon><CopyIcon /></ListItemIcon>
            <ListItemText>复制计划</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => {
            setMenuAnchorEl(null);
          }}>
            <ListItemIcon><ShareIcon /></ListItemIcon>
            <ListItemText>分享计划</ListItemText>
          </MenuItem>
          <Divider />
          <MenuItem 
            onClick={() => {
              if (selectedPlanId) deletePlan(selectedPlanId);
              setMenuAnchorEl(null);
            }}
            sx={{ color: 'error.main' }}
          >
            <ListItemIcon><DeleteIcon color="error" /></ListItemIcon>
            <ListItemText>删除计划</ListItemText>
          </MenuItem>
        </MenuList>
      </Menu>

      {/* 对话框 */}
      {renderFilterDialog()}
      {renderCreateDialog()}

      {/* 通知 */}
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
    </Container>
  );
};

export default TravelPlanList; 