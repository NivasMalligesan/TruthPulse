import React, { useState, useEffect } from 'react'
import { io } from 'socket.io-client'

const Dashboard = () => {
  const [analyses, setAnalyses] = useState([])
  const [inputText, setInputText] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [stats, setStats] = useState({
    total_analyzed: 0,
    suspicious_count: 0,
    avg_credibility: 0,
    trend: 'stable',
    avg_manipulation_index: 0,
    avg_emotional_density: 0,
    fact_checked_claims: 0
  })

  const socket = io('http://localhost:5000')

  useEffect(() => {
    socket.on('new_analysis', (analysis) => {
      setAnalyses(prev => [analysis, ...prev.slice(0, 14)])
      fetchStats()
    })

    fetchStats()
    return () => socket.disconnect()
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats')
      const statsData = await response.json()
      setStats(statsData)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const analyzeText = async () => {
    if (!inputText.trim()) return
    setIsAnalyzing(true)
    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      })
      const analysis = await response.json()
      setAnalyses(prev => [analysis, ...prev.slice(0, 14)])
      setInputText('')
      fetchStats()
    } catch (error) {
      console.error('Analysis failed:', error)
    }
    setIsAnalyzing(false)
  }

  const getScoreColor = (score) => {
    if (score >= 80) return '#10B981'
    if (score >= 65) return '#34D399'
    if (score >= 50) return '#F59E0B'
    if (score >= 35) return '#EF4444'
    return '#DC2626'
  }

  const getScoreBgColor = (score) => {
    if (score >= 80) return '#10B981'
    if (score >= 65) return '#34D399'
    if (score >= 50) return '#F59E0B'
    if (score >= 35) return '#EF4444'
    return '#DC2626'
  }

  const getManipulationColor = (index) => {
    if (index < 1) return '#10B981'
    if (index < 3) return '#F59E0B'
    if (index < 5) return '#EF4444'
    return '#DC2626'
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      analyzeText()
    }
  }

  const formatNumber = (num) => {
    if (!num && num !== 0) return '0'
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M'
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K'
    }
    return num.toString()
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
    }}>
      
      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        padding: '1.5rem 0'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '0 1.5rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{
              width: '3rem',
              height: '3rem',
              background: 'rgba(59, 130, 246, 0.8)',
              borderRadius: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.5rem',
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)'
            }}>🛡️</div>
            <div>
              <h1 style={{
                fontSize: '2rem',
                fontWeight: '800',
                background: 'linear-gradient(135deg, #ffffff, #e5e7eb)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                margin: 0
              }}>TruthPulse AI</h1>
              <p style={{ 
                color: '#BFDBFE', 
                fontSize: '0.875rem', 
                margin: 0
              }}>Next-Gen Fake News Detection with Psychological Analysis</p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              background: 'rgba(16, 185, 129, 0.2)',
              padding: '0.5rem 1rem',
              borderRadius: '2rem',
              border: '1px solid rgba(16, 185, 129, 0.3)'
            }}>
              <div style={{
                width: '0.5rem',
                height: '0.5rem',
                backgroundColor: '#10B981',
                borderRadius: '50%'
              }}></div>
              <span style={{ fontWeight: '600', fontSize: '0.875rem' }}>ADVANCED AI</span>
            </div>
          </div>
        </div>
      </header>

      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '2rem 1.5rem'
      }}>
        
        {/* Enhanced Stats Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          {[
            {
              title: 'Total Analyzed',
              value: stats.total_analyzed,
              trend: 'Real-time processing',
              icon: '📊',
              color: '#3B82F6',
              description: 'Content pieces analyzed'
            },
            {
              title: 'Suspicious Content',
              value: stats.suspicious_count,
              trend: 'Psychological flags',
              icon: '🧠',
              color: '#EF4444',
              description: 'Manipulation detected'
            },
            {
              title: 'Avg Credibility',
              value: `${Math.round(stats.avg_credibility)}%`,
              trend: 'Multi-factor score',
              icon: '🛡️',
              color: getScoreColor(stats.avg_credibility),
              description: 'Enhanced scoring'
            },
            {
              title: 'Manipulation Index',
              value: stats.avg_manipulation_index?.toFixed(1) || '0.0',
              trend: 'Emotion vs Facts',
              icon: '⚖️',
              color: getManipulationColor(stats.avg_manipulation_index || 0),
              description: 'Lower is better'
            }
          ].map((stat, index) => (
            <div 
              key={index}
              style={{
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '1rem',
                padding: '1.5rem',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-5px)'
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0px)'
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <div>
                  <p style={{ 
                    color: '#E5E7EB', 
                    fontSize: '0.875rem', 
                    fontWeight: '600', 
                    margin: '0 0 0.5rem 0'
                  }}>
                    {stat.title}
                  </p>
                  <p style={{ 
                    fontSize: '2.5rem', 
                    fontWeight: '800', 
                    margin: '0 0 0.25rem 0',
                    color: stat.color
                  }}>
                    {stat.value}
                  </p>
                  <p style={{ 
                    color: stat.color, 
                    fontSize: '0.75rem', 
                    margin: 0,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem'
                  }}>
                    <span>📈</span>
                    {stat.trend}
                  </p>
                </div>
                <div style={{
                  padding: '0.75rem',
                  background: `${stat.color}20`,
                  borderRadius: '0.75rem',
                  fontSize: '1.5rem'
                }}>
                  {stat.icon}
                </div>
              </div>
              <p style={{ 
                color: '#9CA3AF', 
                fontSize: '0.75rem', 
                margin: 0,
                borderTop: '1px solid rgba(255,255,255,0.1)',
                paddingTop: '0.75rem'
              }}>
                {stat.description}
              </p>
            </div>
          ))}
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr',
          gap: '2rem'
        }}>
          
          {/* Analysis Input */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '1rem',
            padding: '1.5rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <span style={{ fontSize: '1.5rem' }}>🔍</span>
              <div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: '700', margin: 0 }}>Advanced Content Analysis</h3>
                <p style={{ color: '#BFDBFE', fontSize: '0.875rem', margin: '0.25rem 0 0 0' }}>
                  ML + Psychological Analysis + Style Forensics + Fact Checking
                </p>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Paste any text for multi-dimensional analysis... (Ctrl+Enter to analyze)"
                style={{
                  width: '96%',
                  height: '120px',
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  background: 'rgba(15, 23, 42, 0.8)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  color: 'white',
                  fontSize: '0.875rem',
                  resize: 'none',
                  outline: 'none',
                  fontFamily: 'inherit'
                }}
              />
              <button
                onClick={analyzeText}
                disabled={isAnalyzing || !inputText.trim()}
                style={{
                  width: '100%',
                  background: isAnalyzing || !inputText.trim() 
                    ? 'linear-gradient(135deg, #4B5563, #374151)'
                    : 'linear-gradient(135deg, #2563EB, #7C3AED)',
                  color: 'white',
                  fontWeight: '600',
                  padding: '1rem 1.5rem',
                  borderRadius: '0.75rem',
                  border: 'none',
                  cursor: isAnalyzing || !inputText.trim() ? 'not-allowed' : 'pointer',
                  fontSize: '1rem'
                }}
              >
                {isAnalyzing ? (
                  <>
                    <div style={{
                      width: '1.25rem',
                      height: '1.25rem',
                      border: '2px solid transparent',
                      borderTop: '2px solid white',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                      display: 'inline-block',
                      marginRight: '0.5rem'
                    }}></div>
                    Advanced Analysis in Progress...
                  </>
                ) : (
                  '🚀 Run Multi-Dimensional Analysis'
                )}
              </button>
            </div>
          </div>

          {/* Enhanced Analysis Feed */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '1rem',
            padding: '1.5rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '1.5rem' }}>📊</span>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '700', margin: 0 }}>Advanced Analysis Feed</h3>
                  <p style={{ color: '#BFDBFE', fontSize: '0.875rem', margin: '0.25rem 0 0 0' }}>
                    Real-time multi-dimensional assessment
                  </p>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10B981' }}>
                <div style={{
                  width: '0.5rem',
                  height: '0.5rem',
                  backgroundColor: '#10B981',
                  borderRadius: '50%',
                  animation: 'pulse 2s infinite'
                }}></div>
                <span style={{ fontSize: '0.875rem', fontWeight: '600' }}>AI ACTIVE</span>
              </div>
            </div>
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '1.5rem', 
              maxHeight: '600px', 
              overflowY: 'auto'
            }}>
              {analyses.map((analysis, index) => (
                <div 
                  key={analysis.id} 
                  style={{
                    background: 'rgba(15, 23, 42, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '0.75rem',
                    padding: '1.5rem',
                    borderLeft: `4px solid ${analysis.verdict.color}`,
                    transition: 'all 0.3s ease'
                  }}
                >
                  {/* Main Analysis Header */}
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{
                        width: '2.5rem',
                        height: '2.5rem',
                        background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '1rem',
                        fontWeight: 'bold'
                      }}>
                        🧠
                      </div>
                      <div>
                        <div style={{ fontWeight: '600', color: 'white', fontSize: '1.1rem' }}>
                          Multi-Dimensional Analysis
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>
                          {new Date(analysis.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div style={{
                      background: `${analysis.verdict.color}20`,
                      color: analysis.verdict.color,
                      padding: '0.5rem 1rem',
                      borderRadius: '9999px',
                      fontSize: '0.875rem',
                      fontWeight: '700',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem'
                    }}>
                      {analysis.verdict.icon} {analysis.verdict.status}
                    </div>
                  </div>
                  
                  {/* Content */}
                  <p style={{ 
                    color: '#E5E7EB', 
                    fontSize: '0.9rem', 
                    lineHeight: '1.5rem',
                    marginBottom: '1.5rem',
                    background: 'rgba(255,255,255,0.05)',
                    padding: '1rem',
                    borderRadius: '0.5rem',
                    borderLeft: '3px solid #3B82F6'
                  }}>
                    {analysis.text}
                  </p>
                  
                  {/* Enhanced Metrics Grid */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '1rem',
                    marginBottom: '1.5rem'
                  }}>
                    <div style={{
                      background: 'rgba(16, 185, 129, 0.1)',
                      padding: '0.75rem',
                      borderRadius: '0.5rem',
                      border: '1px solid rgba(16, 185, 129, 0.2)'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: '#10B981', fontWeight: '600', marginBottom: '0.25rem' }}>CREDIBILITY SCORE</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: '800', color: getScoreColor(analysis.credibility_score) }}>
                        {analysis.credibility_score}%
                      </div>
                    </div>
                    
                    <div style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      padding: '0.75rem',
                      borderRadius: '0.5rem',
                      border: '1px solid rgba(59, 130, 246, 0.2)'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: '#3B82F6', fontWeight: '600', marginBottom: '0.25rem' }}>ML CONFIDENCE</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: '800', color: '#3B82F6' }}>
                        {analysis.confidence}%
                      </div>
                    </div>
                    
                    {analysis.enhanced_analysis && (
                      <>
                        <div style={{
                          background: 'rgba(245, 158, 11, 0.1)',
                          padding: '0.75rem',
                          borderRadius: '0.5rem',
                          border: '1px solid rgba(245, 158, 11, 0.2)'
                        }}>
                          <div style={{ fontSize: '0.75rem', color: '#F59E0B', fontWeight: '600', marginBottom: '0.25rem' }}>MANIPULATION INDEX</div>
                          <div style={{ fontSize: '1.5rem', fontWeight: '800', color: getManipulationColor(analysis.enhanced_analysis.manipulation_metrics?.manipulation_index || 0) }}>
                            {analysis.enhanced_analysis.manipulation_metrics?.manipulation_index || 'N/A'}
                          </div>
                        </div>
                        
                        <div style={{
                          background: 'rgba(139, 92, 246, 0.1)',
                          padding: '0.75rem',
                          borderRadius: '0.5rem',
                          border: '1px solid rgba(139, 92, 246, 0.2)'
                        }}>
                          <div style={{ fontSize: '0.75rem', color: '#8B5CF6', fontWeight: '600', marginBottom: '0.25rem' }}>STYLE INCONSISTENCY</div>
                          <div style={{ fontSize: '1.5rem', fontWeight: '800', color: '#8B5CF6' }}>
                            {analysis.enhanced_analysis.style_analysis?.inconsistency_score || 0}%
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Enhanced Analysis Sections */}
                  {analysis.enhanced_analysis && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      {/* Psychological Manipulation */}
                      {analysis.enhanced_analysis.psychological_manipulation && analysis.enhanced_analysis.psychological_manipulation.length > 0 && (
                        <div style={{
                          background: 'rgba(239, 68, 68, 0.1)',
                          border: '1px solid rgba(239, 68, 68, 0.2)',
                          borderRadius: '0.5rem',
                          padding: '1rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                            <span style={{ color: '#EF4444', fontSize: '1.25rem' }}>🧠</span>
                            <h4 style={{ color: '#EF4444', fontSize: '0.9rem', fontWeight: '700', margin: 0 }}>PSYCHOLOGICAL MANIPULATION DETECTED</h4>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {analysis.enhanced_analysis.psychological_manipulation.map((technique, idx) => (
                              <div key={idx} style={{ 
                                display: 'flex', 
                                alignItems: 'center', 
                                gap: '0.5rem',
                                color: '#FCA5A5',
                                fontSize: '0.8rem'
                              }}>
                                <div style={{
                                  width: '0.5rem',
                                  height: '0.5rem',
                                  backgroundColor: '#EF4444',
                                  borderRadius: '50%'
                                }}></div>
                                <span>{technique}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Manipulation Metrics */}
                      {analysis.enhanced_analysis.manipulation_metrics && (
                        <div style={{
                          background: 'rgba(59, 130, 246, 0.1)',
                          border: '1px solid rgba(59, 130, 246, 0.2)',
                          borderRadius: '0.5rem',
                          padding: '1rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                            <span style={{ color: '#3B82F6', fontSize: '1.25rem' }}>📈</span>
                            <h4 style={{ color: '#3B82F6', fontSize: '0.9rem', fontWeight: '700', margin: 0 }}>QUANTITATIVE ANALYSIS</h4>
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '0.75rem', fontSize: '0.8rem' }}>
                            <div>
                              <div style={{ color: '#93C5FD' }}>Emotional Density</div>
                              <div style={{ color: 'white', fontWeight: '600' }}>{(analysis.enhanced_analysis.manipulation_metrics.emotional_density * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div style={{ color: '#93C5FD' }}>Factual Density</div>
                              <div style={{ color: 'white', fontWeight: '600' }}>{(analysis.enhanced_analysis.manipulation_metrics.factual_density * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                              <div style={{ color: '#93C5FD' }}>Sentiment</div>
                              <div style={{ color: 'white', fontWeight: '600' }}>{analysis.enhanced_analysis.manipulation_metrics.sentiment_polarity > 0 ? '😊 Positive' : analysis.enhanced_analysis.manipulation_metrics.sentiment_polarity < 0 ? '😠 Negative' : '😐 Neutral'}</div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Fact Check Results */}
                      {analysis.enhanced_analysis.fact_check && analysis.enhanced_analysis.fact_check.available && (
                        <div style={{
                          background: 'rgba(16, 185, 129, 0.1)',
                          border: '1px solid rgba(16, 185, 129, 0.2)',
                          borderRadius: '0.5rem',
                          padding: '1rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                            <span style={{ color: '#10B981', fontSize: '1.25rem' }}>✅</span>
                            <h4 style={{ color: '#10B981', fontSize: '0.9rem', fontWeight: '700', margin: 0 }}>FACT CHECK VERIFICATION</h4>
                          </div>
                          {analysis.enhanced_analysis.fact_check.results && analysis.enhanced_analysis.fact_check.results.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                              {analysis.enhanced_analysis.fact_check.results.slice(0, 2).map((result, idx) => (
                                <div key={idx} style={{ 
                                  background: 'rgba(255,255,255,0.05)',
                                  padding: '0.5rem',
                                  borderRadius: '0.25rem',
                                  fontSize: '0.8rem'
                                }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ color: '#E5E7EB', fontWeight: '600' }}>Claim: {result.claim.substring(0, 80)}...</span>
                                    <span style={{ 
                                      color: result.rating === 'False' ? '#EF4444' : result.rating === 'True' ? '#10B981' : '#F59E0B',
                                      fontWeight: '700',
                                      background: result.rating === 'False' ? 'rgba(239, 68, 68, 0.2)' : result.rating === 'True' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                                      padding: '0.25rem 0.5rem',
                                      borderRadius: '0.25rem'
                                    }}>
                                      {result.rating}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div style={{ color: '#6B7280', fontSize: '0.8rem' }}>No verifiable claims found for fact checking</div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Traditional Analysis Points */}
                  {analysis.analysis_points && analysis.analysis_points.length > 0 && (
                    <div style={{
                      marginTop: '1rem',
                      padding: '1rem',
                      background: 'rgba(245, 158, 11, 0.1)',
                      border: '1px solid rgba(245, 158, 11, 0.2)',
                      borderRadius: '0.5rem'
                    }}>
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.5rem', 
                        color: '#F59E0B', 
                        fontSize: '0.8rem', 
                        fontWeight: '600',
                        marginBottom: '0.75rem'
                      }}>
                        <span>🚩</span>
                        <span>TRADITIONAL ANALYSIS FLAGS</span>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {analysis.analysis_points.map((point, idx) => (
                          <div key={idx} style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '0.5rem',
                            color: '#FCD34D',
                            fontSize: '0.8rem'
                          }}>
                            <div style={{
                              width: '0.4rem',
                              height: '0.4rem',
                              backgroundColor: '#F59E0B',
                              borderRadius: '50%'
                            }}></div>
                            <span>{point}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              
              {analyses.length === 0 && (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '3rem', 
                  color: '#9CA3AF' 
                }}>
                  <div style={{ fontSize: '4rem', marginBottom: '1rem', opacity: 0.5 }}>🧠</div>
                  <p style={{ fontWeight: '700', marginBottom: '0.5rem', fontSize: '1.2rem' }}>Advanced AI Analysis Ready</p>
                  <p style={{ fontSize: '0.9rem', marginBottom: '2rem' }}>Multi-dimensional fake news detection with psychological analysis</p>
                  <div style={{ 
                    display: 'inline-grid', 
                    gridTemplateColumns: 'repeat(2, 1fr)', 
                    gap: '1rem',
                    background: 'rgba(255,255,255,0.05)',
                    padding: '1.5rem',
                    borderRadius: '1rem',
                    border: '1px solid rgba(255,255,255,0.1)'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#10B981' }}>✅</span>
                      <span style={{ fontSize: '0.8rem' }}>ML Classification</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#3B82F6' }}>🧠</span>
                      <span style={{ fontSize: '0.8rem' }}>Psychological Analysis</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#8B5CF6' }}>📝</span>
                      <span style={{ fontSize: '0.8rem' }}>Style Forensics</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ color: '#F59E0B' }}>🔍</span>
                      <span style={{ fontSize: '0.8rem' }}>Fact Checking</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        ::-webkit-scrollbar {
          width: 8px;
        }
        
        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.3);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.5);
        }
      `}</style>
    </div>
  )
}

export default Dashboard