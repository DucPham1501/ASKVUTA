import { Link } from "react-router";
import { MessageCircle, MapPin, Utensils, Landmark, Calendar, Mountain, Waves, Github, Mail, Linkedin } from "lucide-react";
import { motion } from "motion/react";
import { useRef } from "react";
import Slider from "react-slick";
import "../../styles/carousel.css";

export function HomePage() {
  const newsCarouselRef = useRef<Slider>(null);

  const newsCategories = [
    {
      id: 1,
      category: "Quán ăn ngon",
      title: "Top 10 quán hải sản tươi ngon ở Vũng Tàu",
      description: "Khám phá những quán ăn hải sản được yêu thích nhất tại thành phố biển",
      image: "🦞",
      color: "from-orange-400 to-red-500"
    },
    {
      id: 2,
      category: "Địa điểm du lịch",
      title: "Bãi Sau - Điểm đến lý tưởng cho gia đình",
      description: "Bãi biển đẹp với làn nước trong xanh và cát trắng mịn màng",
      image: "🏖️",
      color: "from-cyan-400 to-blue-500"
    },
    {
      id: 3,
      category: "Văn hóa",
      title: "Lễ hội đình thần Thắng Tam",
      description: "Tìm hiểu về văn hóa tín ngưỡng đặc sắc của người dân Vũng Tàu",
      image: "🎭",
      color: "from-purple-400 to-pink-500"
    },
    {
      id: 4,
      category: "Kinh tế",
      title: "Vũng Tàu phát triển du lịch biển bền vững",
      description: "Các dự án mới nhằm nâng cao chất lượng dịch vụ du lịch",
      image: "📈",
      color: "from-green-400 to-emerald-500"
    },
    {
      id: 5,
      category: "Sự kiện địa phương",
      title: "Festival Biển Vũng Tàu 2026",
      description: "Sự kiện âm nhạc và ẩm thực lớn nhất năm tại bãi Trước",
      image: "🎪",
      color: "from-yellow-400 to-orange-500"
    }
  ];

  const featuredPlaces = [
    {
      name: "Tượng Chúa Kitô",
      type: "Landmark",
      icon: "⛪",
      description: "Biểu tượng nổi tiếng của Vũng Tàu"
    },
    {
      name: "Bãi Trước",
      type: "Beach",
      icon: "🌊",
      description: "Bãi biển sầm uất, nhộn nhịp"
    },
    {
      name: "Núi Lớn",
      type: "Mountain",
      icon: "⛰️",
      description: "Điểm ngắm hoàng hôn tuyệt đẹp"
    },
    {
      name: "Ngọn Hải Đăng",
      type: "Lighthouse",
      icon: "🗼",
      description: "Công trình kiến trúc Pháp cổ"
    }
  ];

  const sliderSettings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 3,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 3000,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 2,
        }
      },
      {
        breakpoint: 640,
        settings: {
          slidesToShow: 1,
        }
      }
    ]
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 via-blue-500 to-indigo-600 opacity-90"></div>
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <pattern id="waves" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
              <path d="M0 50 Q 25 40, 50 50 T 100 50" fill="none" stroke="white" strokeWidth="2" />
              <path d="M0 60 Q 25 50, 50 60 T 100 60" fill="none" stroke="white" strokeWidth="2" />
            </pattern>
            <rect width="100%" height="100%" fill="url(#waves)" />
          </svg>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 sm:py-32">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <div className="flex justify-center mb-6">
              <div className="bg-white/20 backdrop-blur-lg p-4 rounded-3xl shadow-2xl">
                <Waves className="w-16 h-16 text-white" />
              </div>
            </div>
            <h1 className="text-5xl sm:text-6xl font-bold text-white mb-6">
              AskVuta
            </h1>
            <p className="text-xl sm:text-2xl text-cyan-50 mb-8 max-w-3xl mx-auto">
              Khám phá thành phố biển Vũng Tàu cùng AI
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/chat"
                className="inline-flex items-center gap-2 bg-white text-cyan-600 px-8 py-4 rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all"
              >
                <MessageCircle className="w-5 h-5" />
                <span>Chat ngay</span>
              </Link>
              <button className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-md text-white border-2 border-white px-8 py-4 rounded-full hover:bg-white/30 transition-all">
                <MapPin className="w-5 h-5" />
                <span>Khám phá Vũng Tàu</span>
              </button>
            </div>
          </motion.div>
        </div>

        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" className="w-full h-auto">
            <path
              fill="#f0f9ff"
              d="M0,64L48,69.3C96,75,192,85,288,80C384,75,480,53,576,48C672,43,768,53,864,58.7C960,64,1056,64,1152,58.7C1248,53,1344,43,1392,37.3L1440,32L1440,120L1392,120C1344,120,1248,120,1152,120C1056,120,960,120,864,120C768,120,672,120,576,120C480,120,384,120,288,120C192,120,96,120,48,120L0,120Z"
            ></path>
          </svg>
        </div>
      </section>

      {/* About Chatbot Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-sky-50 to-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-sky-900 mb-4">
              Trợ lý AI của bạn
            </h2>
            <p className="text-lg text-sky-700 max-w-2xl mx-auto">
              Tìm hiểu mọi thông tin về Vũng Tàu chỉ với vài câu hỏi
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: MapPin, title: "Du lịch Vũng Tàu", desc: "Gợi ý điểm đến, lịch trình du lịch", color: "from-cyan-500 to-blue-600" },
              { icon: Utensils, title: "Đặc sản địa phương", desc: "Món ăn đặc trưng và quán ăn ngon", color: "from-orange-500 to-red-600" },
              { icon: Landmark, title: "Văn hóa & lịch sử", desc: "Tìm hiểu về lịch sử và văn hóa", color: "from-purple-500 to-pink-600" },
              { icon: Calendar, title: "Tin tức thành phố", desc: "Cập nhật tin tức và sự kiện mới", color: "from-green-500 to-emerald-600" }
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -8, scale: 1.02 }}
                className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all border border-cyan-100"
              >
                <div className={`bg-gradient-to-br ${item.color} w-14 h-14 rounded-xl flex items-center justify-center mb-4 shadow-md`}>
                  <item.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-sky-900 mb-2">{item.title}</h3>
                <p className="text-sky-700">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* News & Information Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-sky-900 mb-4">
              Tin tức & Thông tin
            </h2>
            <p className="text-lg text-sky-700">
              Cập nhật mới nhất về Vũng Tàu
            </p>
          </motion.div>

          <div className="news-carousel">
            <Slider ref={newsCarouselRef} {...sliderSettings}>
              {newsCategories.map((news) => (
                <div key={news.id} className="px-3">
                  <motion.div
                    whileHover={{ y: -5 }}
                    className="bg-white rounded-2xl shadow-lg overflow-hidden border border-cyan-100 h-full"
                  >
                    <div className={`h-40 bg-gradient-to-br ${news.color} flex items-center justify-center text-6xl`}>
                      {news.image}
                    </div>
                    <div className="p-6">
                      <span className="text-sm font-medium text-cyan-600 uppercase tracking-wide">
                        {news.category}
                      </span>
                      <h3 className="text-xl font-semibold text-sky-900 mt-2 mb-3">
                        {news.title}
                      </h3>
                      <p className="text-sky-700 mb-4">
                        {news.description}
                      </p>
                      <button className="text-cyan-600 hover:text-cyan-700 font-medium flex items-center gap-2 group">
                        Đọc thêm
                        <span className="group-hover:translate-x-1 transition-transform">→</span>
                      </button>
                    </div>
                  </motion.div>
                </div>
              ))}
            </Slider>
          </div>
        </div>
      </section>

      {/* Featured Places Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-cyan-50 to-blue-50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-sky-900 mb-4">
              Điểm đến nổi bật
            </h2>
            <p className="text-lg text-sky-700">
              Những địa điểm không thể bỏ qua khi đến Vũng Tàu
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {featuredPlaces.map((place, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ scale: 1.05, rotate: 1 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg hover:shadow-2xl transition-all cursor-pointer border border-cyan-100"
              >
                <div className="text-6xl mb-4 text-center">{place.icon}</div>
                <h3 className="text-xl font-semibold text-sky-900 text-center mb-2">
                  {place.name}
                </h3>
                <p className="text-sm text-cyan-600 text-center mb-2">{place.type}</p>
                <p className="text-sky-700 text-center text-sm">
                  {place.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Founder Section */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-white">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="bg-gradient-to-br from-cyan-50 to-blue-50 rounded-3xl p-8 sm:p-12 shadow-xl border border-cyan-200"
          >
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full mb-6 shadow-lg">
                <span className="text-3xl text-white">🚀</span>
              </div>
              <h2 className="text-3xl font-bold text-sky-900 mb-4">Về dự án</h2>
              <p className="text-lg text-sky-700 mb-6 max-w-2xl mx-auto">
                AskVuta là dự án chatbot thông minh giúp du khách và người dân
                dễ dàng tìm kiếm thông tin về thành phố biển Vũng Tàu.
              </p>
              <div className="flex flex-wrap justify-center gap-4 mb-8">
                <div className="bg-white rounded-xl px-6 py-3 shadow-md">
                  <p className="text-sm text-sky-600">Mục tiêu</p>
                  <p className="font-semibold text-sky-900">Hỗ trợ du lịch thông minh</p>
                </div>
                <div className="bg-white rounded-xl px-6 py-3 shadow-md">
                  <p className="text-sm text-sky-600">Công nghệ</p>
                  <p className="font-semibold text-sky-900">AI & React</p>
                </div>
              </div>
              <div className="flex justify-center gap-4">
                <a href="https://github.com" target="_blank" rel="noopener noreferrer"
                   className="flex items-center gap-2 bg-sky-900 text-white px-6 py-3 rounded-full hover:bg-sky-800 transition-colors shadow-md">
                  <Github className="w-5 h-5" />
                  <span>GitHub</span>
                </a>
                <a href="mailto:contact@example.com"
                   className="flex items-center gap-2 bg-cyan-500 text-white px-6 py-3 rounded-full hover:bg-cyan-600 transition-colors shadow-md">
                  <Mail className="w-5 h-5" />
                  <span>Email</span>
                </a>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gradient-to-br from-sky-900 to-blue-950 text-white py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Waves className="w-6 h-6" />
                <span className="font-semibold text-lg">AskVuta</span>
              </div>
              <p className="text-cyan-200">
                Trợ lý thông minh cho thành phố biển Vũng Tàu
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Liên kết</h4>
              <ul className="space-y-2 text-cyan-200">
                <li><Link to="/" className="hover:text-white transition-colors">Trang chủ</Link></li>
                <li><Link to="/chat" className="hover:text-white transition-colors">Chatbot</Link></li>
                <li><a href="#" className="hover:text-white transition-colors">Về chúng tôi</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Liên hệ</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Kết nối</h4>
              <div className="flex gap-3">
                <a href="#" className="bg-white/10 p-3 rounded-full hover:bg-white/20 transition-colors">
                  <Github className="w-5 h-5" />
                </a>
                <a href="#" className="bg-white/10 p-3 rounded-full hover:bg-white/20 transition-colors">
                  <Mail className="w-5 h-5" />
                </a>
                <a href="#" className="bg-white/10 p-3 rounded-full hover:bg-white/20 transition-colors">
                  <Linkedin className="w-5 h-5" />
                </a>
              </div>
            </div>
          </div>
          <div className="border-t border-cyan-800 pt-8 text-center text-cyan-300">
            <p>&copy; 2026 Vung Tau AI Assistant. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
