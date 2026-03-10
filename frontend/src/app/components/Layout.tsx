import { Outlet, Link, useLocation } from "react-router";
import { Waves, MessageCircle, Home } from "lucide-react";

export function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-cyan-50 to-blue-100">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 backdrop-blur-md bg-white/80 border-b border-cyan-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
              <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl shadow-lg">
                <Waves className="w-6 h-6 text-white" />
              </div>
              <span className="font-semibold text-sky-900">AskVuta</span>
            </Link>

            <div className="flex gap-4">
              <Link
                to="/"
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  location.pathname === "/"
                    ? "bg-cyan-500 text-white shadow-md"
                    : "text-sky-700 hover:bg-cyan-100"
                }`}
              >
                <Home className="w-4 h-4" />
                <span>Trang chủ</span>
              </Link>
              <Link
                to="/chat"
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  location.pathname === "/chat"
                    ? "bg-cyan-500 text-white shadow-md"
                    : "text-sky-700 hover:bg-cyan-100"
                }`}
              >
                <MessageCircle className="w-4 h-4" />
                <span>Chatbot</span>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <Outlet />
    </div>
  );
}
