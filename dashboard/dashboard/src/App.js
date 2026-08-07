import React, { useState, useEffect } from "react";
import {
  Layout,
  Typography,
  Select,
  Input,
  Button,
  Card,
  Row,
  Col,
  Tag,
  Statistic,
  Progress,
  Alert,
  Empty,
} from "antd";

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

const API_BASE = "http://127.0.0.1:8000";

/*const TOPIC_LABELS = {
  0: "Topic 0 — Islamic Culture",
  1: "Topic 1 — West Africa Politics",
  2: "Topic 2 — Religious Community",
};
*/
function getColor(level) {
  if (level === "HIGH")   return "#ff4d4f";
  if (level === "MEDIUM") return "#faad14";
  return "#52c41a";
}

export default function App() {
  const [query, setQuery]   = useState("");
  const [topicId, setTopicId] = useState(null);
  const [result, setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(null);
  const [topics, setTopics]  = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/topics`)
      .then((r) => r.json())
      .then((data) => setTopics(data.available_topic_ids || []))
      .catch(() => setError("Could not connect to API. Make sure the backend is running."));
  }, []);

  const handleSearch = async () => {
    if (!query.trim() || topicId === null) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/detect`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query: query.trim(), topic_id: topicId }),
      });
      if (!response.ok) throw new Error("API error: " + response.status);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout style={{ minHeight: "100vh", background: "#f0f2f5" }}>

      <Header style={{ background: "#001529", padding: "0 32px", display: "flex", alignItems: "center" }}>
        <Title level={4} style={{ color: "#fff", margin: 0 }}>
          PSemQE — Ambiguity Detection
        </Title>
      </Header>

      <Content style={{ padding: "32px 48px" }}>

        {error && (
          <Alert message={error} type="error" showIcon closable
            style={{ marginBottom: 24 }} onClose={() => setError(null)} />
        )}

        {/* Search Card */}
        <Card title="Query Search" style={{ marginBottom: 24 }}>
          <Row gutter={16} align="middle">
            <Col span={12}>
              <Text type="secondary">Search Query</Text>
              <Input
                size="large"
                placeholder='e.g. "Amazon", "Python", "Java"'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onPressEnter={handleSearch}
                style={{ marginTop: 4 }}
              />
            </Col>
            <Col span={8}>
              <Text type="secondary">User Profile (LDA Topic)</Text>
              <Select
                size="large"
                placeholder="Select a topic profile"
                style={{ width: "100%", marginTop: 4 }}
                onChange={(val) => setTopicId(val)}
              >
                {topics.map((id) => (
                  <Option key={id} value={id}>
                    Topic {id}
                  </Option>
                ))}
              </Select>
            </Col>
            <Col span={4} style={{ paddingTop: 22 }}>
              <Button
                type="primary" size="large" block
                loading={loading} onClick={handleSearch}
                disabled={!query.trim() || topicId === null}
              >
                Detect
              </Button>
            </Col>
          </Row>
        </Card>

        {result ? (
          <Row gutter={24}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Base Ambiguity Score"
                  value={result.base_score}
                  precision={4}
                />
                <Progress
                  percent={Math.round(result.base_score * 100)}
                  style={{ marginTop: 12 }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Profile Matching Score"
                  value={parseFloat((result.final_score - result.base_score).toFixed(4))}
                  precision={4}
                  valueStyle={{ color: (result.final_score - result.base_score) < 0 ? "#52c41a" : "#ff4d4f" }}
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {(result.final_score - result.base_score) < 0
                    ? "Profile reduced ambiguity"
                    : "Profile increased ambiguity"}
                </Text>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Final Ambiguity Score"
                  value={result.final_score}
                  precision={4}
                  valueStyle={{ color: getColor(result.ambiguity_level) }}
                />
                <Progress
                  percent={Math.round(result.final_score * 100)}
                  strokeColor={getColor(result.ambiguity_level)}
                  style={{ marginTop: 12 }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card style={{ textAlign: "center" }}>
                <Text type="secondary">Ambiguity Level</Text>
                <div style={{ marginTop: 16 }}>
                  <Tag
                    color={getColor(result.ambiguity_level)}
                    style={{ fontSize: 16, padding: "4px 16px" }}
                  >
                    {result.ambiguity_level}
                  </Tag>
                </div>
                <div style={{ marginTop: 12 }}>
                  <Tag color={result.is_ambiguous ? "red" : "green"}>
                    {result.is_ambiguous ? "Ambiguous Query" : "Unambiguous Query"}
                  </Tag>
                </div>
              </Card>
            </Col>
          </Row>

        //wordnet results
          
        ) : (
          <Card>
            <Empty description="Enter a query and select a user profile to detect ambiguity" />
          </Card>
        )}
        {result && (
  <Row gutter={24} style={{ marginTop: 24 }}>
    <Col span={24}>
      <Card title="WordNet Valuation">
        <Row gutter={24}>
          <Col span={8}>
            <Statistic
              title="Average Synsets per Word"
              value={result.wordnet_avg_synsets}
              precision={2}
            />
          </Col>
          <Col span={8}>
            <Text type="secondary">WordNet Label</Text>
            <div style={{ marginTop: 8 }}>
              <Tag
                color={result.wordnet_is_ambiguous ? "red" : "green"}
                style={{ fontSize: 14, padding: "4px 12px" }}
              >
                {result.wordnet_label}
              </Tag>
            </div>
          </Col>
          <Col span={8}>
            <Text type="secondary">Model vs WordNet</Text>
            <div style={{ marginTop: 8 }}>
              <Tag
                color={result.is_ambiguous === result.wordnet_is_ambiguous ? "green" : "orange"}
                style={{ fontSize: 14, padding: "4px 12px" }}
              >
                {result.is_ambiguous === result.wordnet_is_ambiguous
                  ? "✓ Model agrees with WordNet"
                  : "≠ Model disagrees with WordNet"}
              </Tag>
            </div>
          </Col>
        </Row>
      </Card>
    </Col>
  </Row>
)}

      </Content>

      <Footer style={{ textAlign: "center", color: "#8c8c8c" }}>
        PSemQE — Ambiguity Detection · University of Bayreuth
      </Footer>

    </Layout>
  );
}